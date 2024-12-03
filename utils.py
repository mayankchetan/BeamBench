# utilities for generating beams for benchmarking

import numpy as np
import matplotlib.pyplot as plt
import warnings
from functools import reduce
import operator
import os
import copy

import welib.fast.beamdyn as bd
from welib.FEM.fem_beam import *
from welib.beams.theory import *

import ruamel.yaml as ry


# try to import sectionproperties
try:
    from sectionproperties.pre.library import circular_section, rectangular_section, circular_hollow_section, rectangular_hollow_section
    from sectionproperties.pre import CompoundGeometry
    from sectionproperties.analysis import Section

    sp_lib = True

    def sectionProp_SectionAnalysis(geometry, plot=False, mesh_size=0):
        sec = Section(geometry=geometry.create_mesh(mesh_sizes=[mesh_size]))
        sec.calculate_geometric_properties()
        sec.calculate_warping_properties()
        sec.calculate_plastic_properties()

        if plot:
            geometry.plot_geometry()
            sec.plot_mesh()
            plt.show()

        return sec
    
    def extractSecProps(sec, getShearCoeff = False, v=0.33):
        # Easy parameters to extract
        area = sec.get_area()
        ic = sec.get_ic() # inertia about centroid
        ig = sec.get_ig() # inertia about global axis for the section properties library
        j = sec.get_j()

        if getShearCoeff:
            warnings.warn("MC: Shear coefficients calculations are sketchy, read theory", UserWarning)

            # https://sectionproperties.readthedocs.io/en/stable/user_guide/theory.html#shear-centre for delta_s

            delta_s = 2 * (1 + v) *  (ic[0]*ic[1] - ic[2]**2) # can we still use this for principal axes?

            # https://sectionproperties.readthedocs.io/en/stable/gen/sectionproperties.analysis.section.Section.html#sectionproperties.analysis.section.Section.get_as_p
            a_s = sec.get_as_p() 


            # from https://sectionproperties.readthedocs.io/en/stable/user_guide/theory.html#shear-deformation-coefficients
            kx = (delta_s**2) / (a_s[0])
            ky = (delta_s**2) / (a_s[1])


            return area, ic[0], ic[1], ig[0], ig[1], j, kx, ky

        return area, ic[0], ic[1], ig[0], ig[1], j



except ImportError:
    Warning('sectionproperties not installed, airfoil shapes will not be available')
    sp_lib = False
    pass

# helper functions

def remove_numpy(fst_vt):
    # recursively move through nested dictionary, remove numpy data types
    # for formatting dictionaries before writing to yaml files
    # from https://github.com/WISDEM/WEIS/blob/main/weis/aeroelasticse/FileTools.py#L19

    def get_dict(vartree, branch):
        return reduce(operator.getitem, branch, vartree)

    def loop_dict(vartree, branch):
        if type(vartree) is not dict:
            return fst_vt
        for var in vartree.keys():
            branch_i = copy.copy(branch)
            branch_i.append(var)
            if type(vartree[var]) is dict:
                loop_dict(vartree[var], branch_i)
            else:
                data_type = type(get_dict(fst_vt, branch_i[:-1])[branch_i[-1]])

                if data_type in [np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]:
                    get_dict(fst_vt, branch_i[:-1])[branch_i[-1]] = int(get_dict(fst_vt, branch_i[:-1])[branch_i[-1]])
                elif data_type in [np.single, np.double, np.longdouble, np.csingle, np.cdouble, np.float16, np.float32, np.float64, np.complex64, np.complex128]:
                    get_dict(fst_vt, branch_i[:-1])[branch_i[-1]] = float(get_dict(fst_vt, branch_i[:-1])[branch_i[-1]])
                elif data_type in [np.bool_]:
                    get_dict(fst_vt, branch_i[:-1])[branch_i[-1]] = bool(get_dict(fst_vt, branch_i[:-1])[branch_i[-1]])
                elif data_type in [np.ndarray]:
                    get_dict(fst_vt, branch_i[:-1])[branch_i[-1]] = get_dict(fst_vt, branch_i[:-1])[branch_i[-1]].tolist()
                elif data_type in [list,tuple]:
                    for item in get_dict(fst_vt, branch_i[:-1])[branch_i[-1]]:
                        remove_numpy(item)

    # set fast variables to update values
    loop_dict(fst_vt, [])

    return fst_vt

def save_yaml(outdir, fname, data_out):
    # from https://github.com/WISDEM/WEIS/blob/main/weis/aeroelasticse/FileTools.py#L123

    if not os.path.isdir(outdir) and outdir!='':
        os.makedirs(outdir)
    fname = os.path.join(outdir, fname)

    data_out = remove_numpy(data_out)

    f = open(fname, "w")
    yaml=ry.YAML()
    yaml.default_flow_style = None
    yaml.width = float("inf")
    yaml.indent(mapping=4, sequence=6, offset=3)
    yaml.dump(data_out, f)
    f.close()

class Shape:
    """Base class for all cross-sectional shapes"""
    def __init__(self):
        self.area = None
        self.Ixx = None
        self.Iyy = None
        self.J = None
        self.kx = None
        self.ky = None

        # 0-origin based on BeamDyn Theory
        self.origin = (0, 0)
        self.sp2origin = (0, 0) # vector from section properties origin to the BeamDyn origin

        # BeamDyn properties
        self.KK = None
        self.MM = None
        
        # SectionProperties results
        self.area_sp = None
        self.Ixx_sp = None
        self.Iyy_sp = None
        self.J_sp = None
        self.kx_sp = None
        self.ky_sp = None
        self.section = None

        # Frequency and mode shapes nested dictionary
        self.freqModes = {'FEM': {}, 
                        'Theory': {}}
        
    @property
    def saveFolder(self):
        return self._saveFolder


    @saveFolder.setter
    def saveFolder(self, folder):
        self._saveFolder = folder
        if not os.path.isdir(folder):
            os.makedirs(folder)

    def getBeamDynMats(self, E, G, Nu, rho, secProp = True):
        # returns the stiffness and mass matrices for the crossection using beamdyn.py from welib
        # E: Young's modulus, G: Shear modulus, Nu: Poisson's ratio, rho: density
        # Choose properties based on secProp flag

        # Overide secProp if set to True and sectionproperties is not installed
        secProp = secProp and sp_lib

        # adding the material properties to the class
        self.E = E
        self.G = G
        self.Nu = Nu
        self.rho = rho

        area = self.area_sp if secProp else self.area
        Ixx = self.Ixx_sp if secProp else self.Ixx
        Iyy = self.Iyy_sp if secProp else self.Iyy
        Ip = Ixx + Iyy
        J = self.J_sp if secProp else self.J
        kx = self.kx_sp if secProp else self.kx
        ky = self.ky_sp if secProp else self.ky

        # this will be in the section properties origin
        elasticCentroid = self.section.get_c() if secProp else (0, 0)
        # convert to BeamDyn origin
        x_C = elasticCentroid[0] - self.sp2origin[0] 
        y_C = elasticCentroid[1] - self.sp2origin[1] 
        theta_p = self.section.get_phi() if secProp else 0

        shearCenter = self.section.get_sc() if secProp else (0, 0)
        # convert to BeamDyn origin
        x_S = shearCenter[0] - self.sp2origin[0]
        y_S = shearCenter[1] - self.sp2origin[1]
        theta_s = 0

        # for mass properties assuming CG is same as centroid since unform mass distribution
        x_G = x_C
        y_G = y_C
        theta_i = 0 # due to symmertry of the sections implimented here

# def KK(EA, EI_x, EI_y, GKt, GA, kxs, kys, x_C=0, y_C=0, theta_p=0, x_S=0, y_S=0, theta_s=0):
        self.KK = bd.KK(
                        EA = E * area,
                        EI_x = E * Ixx,
                        EI_y = E * Iyy,
                        GKt = G * J,
                        GA = G * area,
                        kxs = kx,
                        kys = ky,
                        x_C = x_C,
                        y_C = y_C,
                        theta_p = theta_p,
                        x_S = x_S,
                        y_S = y_S,
                        theta_s = theta_s
                    )

# def MM(m,I_x,I_y,I_p,x_G=0,y_G=0,theta_i=0):
        self.MM = bd.MM(
                        m = rho * area, 
                        I_x = Ixx,
                        I_y = Iyy,
                        I_p = Ip,
                        x_G = x_G,
                        y_G = y_G,
                        theta_i = theta_i
                    )
        return self.KK, self.MM
    
    # Using cbeam and other functions from welib to extract expected frequency and mode shapes
    def FreqAndModes(self,Length, nel=None, BC='clamped-free', element='frame3d', M_tip=None, secProp = True, nModes=10, plot=True):
        # check if MM and KK are computed
        if self.MM.all == None or self.KK.all == None:
            warnings.warn("MM and KK matrices are not computed, use getBeamDynMats() to compute them", UserWarning)
            return None, None, None

        # verify if section properties is installed
        secProp = secProp and sp_lib 

        # --- Compute FEM model and mode shapes
        FEM=cbeam(Length,
                    m = (self.area_sp if secProp else self.area) * self.rho,
                    EIx = self.E * (self.Ixx_sp if secProp else self.Ixx),
                    EIy = self.E * (self.Iyy_sp if secProp else self.Iyy),
                    EIz = self.E * ((self.Ixx_sp if secProp else self.Ixx) + (self.Iyy_sp if secProp else self.Iyy)),
                    EA = self.E * (self.area_sp if secProp else self.area),
                    A = (self.area_sp if secProp else self.area),
                    E = self.E,
                    G = self.G,
                    Kt = (self.J_sp if secProp else self.J),
                    element = element, 
                    nel = nel, 
                    BC = BC, 
                    M_tip = M_tip)
        
        # Extract the first nModes mode shapes and arrange them in a dictionary
        # cbeam assumes x along blade length, y along flapwise, z along edgewise

        # Extracting the mode shapes based on common names
        for idx,modeName in enumerate(FEM['modeNames']):
            if modeName.startswith('uy'):
                modeNameEasy = f'flapwise_{modeName[-1]}'
                self.freqModes['FEM'][modeNameEasy] = {}
                self.freqModes['FEM'][modeNameEasy]['xNodes'] = FEM['xNodes'][0,:] # assuming non curved beam
                self.freqModes['FEM'][modeNameEasy]['modeShape'] = FEM['Q'][1::6, idx]
                self.freqModes['FEM'][modeNameEasy]['frequency'] = FEM['freq'][idx]

            if modeName.startswith('uz'):
                modeNameEasy = f'edgewise_{modeName[-1]}'
                self.freqModes['FEM'][modeNameEasy] = {}
                self.freqModes['FEM'][modeNameEasy]['xNodes'] = FEM['xNodes'][0,:]
                self.freqModes['FEM'][modeNameEasy]['modeShape'] = FEM['Q'][2::6, idx]
                self.freqModes['FEM'][modeNameEasy]['frequency'] = FEM['freq'][idx]

            if modeName.startswith('vx'):
                modeNameEasy = f'torsional_{modeName[-1]}'
                self.freqModes['FEM'][modeNameEasy] = {}
                self.freqModes['FEM'][modeNameEasy]['xNodes'] = FEM['xNodes'][0,:]
                self.freqModes['FEM'][modeNameEasy]['modeShape'] = FEM['Q'][3::6, idx]
                self.freqModes['FEM'][modeNameEasy]['frequency'] = FEM['freq'][idx]

        # --- Theory based on the beam bending modes - Falpwise
        t_freq, t_x, t_q,_,_ = UniformBeamBendingModes(Type = 'unloaded-clamped-free',
                                         EI = self.E * (self.Ixx_sp if secProp else self.Ixx),
                                         rho = self.rho,
                                         A = (self.area_sp if secProp else self.area),
                                         L = Length,
                                         nModes = 3)
        
        self.assignTheory(t_freq, t_x, t_q, 'flapwise')
        
        t_freq, t_x, t_q,_,_ = UniformBeamBendingModes(Type = 'unloaded-clamped-free',
                                            EI = self.E * (self.Iyy_sp if secProp else self.Iyy),
                                            rho = self.rho,
                                            A = (self.area_sp if secProp else self.area),
                                            L = Length,
                                            nModes = 3)
        
        self.assignTheory(t_freq, t_x, t_q, 'edgewise')
        
        t_freq, t_x, t_q,_ = UniformBeamTorsionModes(Type = 'unloaded-clamped-free',
                                                G = self.G,
                                                Kt = self.J_sp if secProp else self.J,
                                                Ip = (self.Ixx_sp if secProp else self.Ixx) + (self.Iyy_sp if secProp else self.Iyy),
                                                rho = self.rho,
                                                A = (self.area_sp if secProp else self.area),
                                                L = Length,
                                                nModes = 3)

        self.assignTheory(t_freq, t_x, t_q, 'torsional')

        if plot:

            if self.saveFolder == None:
                self.saveFolder = os.getcwd()

            self.plotFreqAndModes(mode='flapwise', mNos=np.min((sum('uz' in x for x in FEM['modeNames']),3)))
            self.plotFreqAndModes(mode='edgewise', mNos=np.min((sum('uy' in x for x in FEM['modeNames']),3)))
            self.plotFreqAndModes(mode='torsional', mNos=np.min((sum('vx' in x for x in FEM['modeNames']),3)))

        save_yaml(self.saveFolder, f'{self.__class__.__name__}_BeamFreqModes.yaml', self.freqModes)

        return self.freqModes

    def assignTheory(self, freqs, x, q, mode):

        for idx, freq in enumerate(freqs):
            modeName = f'{mode}_{idx+1}'
            self.freqModes['Theory'][modeName] = {}
            self.freqModes['Theory'][modeName]['xNodes'] = x
            self.freqModes['Theory'][modeName]['frequency'] = freq
            self.freqModes['Theory'][modeName]['modeShape'] = q[idx, :]



    def plotFreqAndModes(self, mode='flapwise', mNos = 1, FEM=True, Theory=True):
        # plot the frequency and mode shapes
        fig,ax = plt.subplots(mNos, 1, sharex = True, figsize=(6.4,4.8*mNos)) # (6.4,4.8)

        for i in range(mNos):
            
            modeName = f'{mode}_{i+1}'

            if mNos == 1:
                if FEM:
                    ax.plot(self.freqModes['FEM'][modeName]['xNodes'], self.freqModes['FEM'][modeName]['modeShape'], 'b-', label=f'FEM @ {self.freqModes["FEM"][modeName]["frequency"]:.2f} Hz')
                if Theory:
                    ax.plot(self.freqModes['Theory'][modeName]['xNodes'], self.freqModes['Theory'][modeName]['modeShape'], 'k--', label=f'Theory @ {self.freqModes["Theory"][modeName]["frequency"]:.2f} Hz')
            
            else:
                if FEM:
                    ax[i].plot(self.freqModes['FEM'][modeName]['xNodes'], self.freqModes['FEM'][modeName]['modeShape'], 'b-', label=f'FEM @ {self.freqModes["FEM"][modeName]["frequency"]:.2f} Hz')
                if Theory:
                    ax[i].plot(self.freqModes['Theory'][modeName]['xNodes'], self.freqModes['Theory'][modeName]['modeShape'], 'k--', label=f'Theory @ {self.freqModes["Theory"][modeName]["frequency"]:.2f} Hz')
                ax[i].set_ylabel('Deflection [-]')
                ax[i].set_title(f'{mode} {i+1} of beam')
                ax[i].legend()

        if mNos == 1:
            ax.set_xlabel('Beam span [-]')
            ax.set_ylabel('Deflection [-]')
            ax.set_title(f'{mode} of beam')
            ax.legend()

        else:
            ax[-1].set_xlabel('Beam span [-]')


        plt.savefig(f'{self.saveFolder}/{mode}_mode_shapes.png')
        plt.close()


class circle(Shape):
    def __init__(self, r, plot=False, mesh_size=0, v=0.33):
        super().__init__()
        self.r = r

        self.area = np.pi*self.r**2
        self.Ixx = np.pi*self.r**4/4
        self.Iyy = np.pi*self.r**4/4
        self.J = np.pi*self.r**4/2

       # shear coeeficients from https://en.wikipedia.org/wiki/Timoshenko–Ehrenfest_beam_theory
        self.kx = (6 * (1 + v)) / (7 + 6 * v)
        self.ky = self.kx

        # Origin info
        self.origin = (0, 0)
        self.sp2origin = (0, 0) # https://sectionproperties.readthedocs.io/en/stable/gen/sectionproperties.pre.library.primitive_sections.circular_section.html

        if sp_lib:
            self.section = sectionProp_SectionAnalysis(geometry=circular_section(d=self.r*2, n=200), plot=plot, mesh_size=mesh_size)
            # setting Ixx & Iyy to centroidal axis for symmetry, therfore being about BeamDyn 0-origin
            self.area_sp, self.Ixx_sp, self.Iyy_sp, _, _, self.J_sp, self.kx_sp, self.ky_sp = extractSecProps(self.section, getShearCoeff=True, v=v)  


class rectangle(Shape):
    def __init__(self, b, h, plot=False, mesh_size=0, v=0.33):
        super().__init__()
        # b: width or in x-axis, h: height, or in y-axis
        self.b = b
        self.h = h
        # setting a to be the long side, and b to be the short side
        self.aa, self.bb = max(b, h), min(b, h)

        self.area = self.b*self.h
        self.Ixx = self.b*self.h**3/12
        self.Iyy = self.b**3*self.h/12
        # based on St. Venant's torsion constant; Roark's Formulas for Stress and Strain, 6th ed., Table 20, Case 4
        self.J = (self.aa * self.bb**3 / 16) * (16/3 - 3.36*(self.bb/self.aa)*(1-self.bb**4/(12*self.aa**4)))

        # shear coeeficients from https://en.wikipedia.org/wiki/Timoshenko–Ehrenfest_beam_theory
        self.kx = (10*(1+v)) / (12+11*v)
        self.ky = self.kx # independant of aspect ratio, not sure why -> The Shear Coefficient in Timoshenko's BeamTheory by G R Cowper, 1966

        # Origin info
        self.origin = (0, 0)
        self.sp2origin =  (b/2, h/2)# https://sectionproperties.readthedocs.io/en/stable/gen/sectionproperties.pre.library.primitive_sections.rectangular_section.html


        if sp_lib:
            self.section = sectionProp_SectionAnalysis(geometry=rectangular_section(b=self.b, d=self.h), plot=plot, mesh_size=mesh_size)
            # setting Ixx & Iyy to centroidal axis for symmetry, therfore being about BeamDyn 0-origin
            self.area_sp, self.Ixx_sp, self.Iyy_sp, _, _, self.J_sp, self.kx_sp, self.ky_sp = extractSecProps(self.section, getShearCoeff=True, v=v)  


class hollowCircle(Shape):
    def __init__(self, r, t, plot=False, mesh_size=0, v=0.33):
        super().__init__()
        # r: outer radius, t: thickness
        self.r = r
        self.t = t

        # check if the thickness is valid
        if t >= r:
            raise ValueError('Invalid thickness, must be less than the radius')

        self.circle = circle(r)
        self.innerCircle = circle(r-t)

        self.area = self.circle.area - self.innerCircle.area
        self.Ixx = self.circle.Ixx - self.innerCircle.Ixx
        self.Iyy = self.circle.Iyy - self.innerCircle.Iyy
        self.J = self.circle.J - self.innerCircle.J

        # origin info
        self.origin = (0, 0)
        self.sp2origin = (0, 0) # https://sectionproperties.readthedocs.io/en/stable/gen/sectionproperties.pre.library.steel_sections.circular_hollow_section.html#sectionproperties.pre.library.steel_sections.circular_hollow_section

        if sp_lib:
            self.section = sectionProp_SectionAnalysis(geometry=circular_hollow_section(d=self.r*2, t=self.t, n=200), plot=plot, mesh_size=mesh_size)
            # setting Ixx & Iyy to centroidal axis for symmetry, therfore being about BeamDyn 0-origin
            self.area_sp, self.Ixx_sp, self.Iyy_sp, _, _, self.J_sp, self.kx_sp, self.ky_sp = extractSecProps(self.section, getShearCoeff=True, v=v)  

    @property
    def kx(self):
        raise NotImplementedError("Shear coefficient kx is not implemented for hollow circular sections, use kx_sp instead")
    
    @kx.setter
    def kx(self, value):
        self._kx = None

    @property 
    def ky(self):
        raise NotImplementedError("Shear coefficient ky is not implemented for hollow circular sections, use kx_sp instead")
    
    @ky.setter
    def ky(self, value):
        self._ky = None



class hollowRectangle(Shape):
    def __init__(self, b, h, t, plot=False, mesh_size=0, v=0.33):
        super().__init__()
        # b: width or in x-axis, h: height, or in y-axis, t: thickness
        self.b = b
        self.h = h
        self.t = t

        # check if the thickness is valid
        if t >= b or t >= h:
            raise ValueError('Invalid thickness, must be less than both dimensions')

        # setting a to be the long side, and b to be the short side
        self.aa, self.bb = max(b, h), min(b, h)

        self.area = self.b*self.h - (self.b-2*self.t)*(self.h-2*self.t)
        self.Ixx = self.b*self.h**3/12 - (self.b-2*self.t)*(self.h-2*self.t)**3/12
        self.Iyy = self.b**3*self.h/12 - (self.b-2*self.t)**3*(self.h-2*self.t)/12
        # based on St. Venant's torsion constant; Roark's Formulas for Stress and Strain, 6th ed., Table 20, Case 16
        # Warning: this torsion constant approximation can have high error (>10%) for some geometries
        self._J = (2*self.t*self.t* ((self.aa - self.t)**2) * ((self.bb - self.t)**2)) / ((self.aa * self.t) + (self.bb * self.t) - (2 * self.t * self.t))

        # origin info
        self.origin = (0, 0)
        self.sp2origin =  (b/2, h/2) # https://sectionproperties.readthedocs.io/en/stable/gen/sectionproperties.pre.library.steel_sections.rectangular_hollow_section.html


        if sp_lib:
            self.section = sectionProp_SectionAnalysis(geometry=rectangular_hollow_section(b=self.b, d=self.h, t=self.t, r_out=0, n_r=1), plot=plot, mesh_size=mesh_size)
            # setting Ixx & Iyy to centroidal axis for symmetry, therfore being about BeamDyn 0-origin
            self.area_sp, self.Ixx_sp, self.Iyy_sp, _, _, self.J_sp, self.kx_sp, self.ky_sp = extractSecProps(self.section, getShearCoeff=True, v=v)  

    @property
    def J(self):
        warnings.warn("The torsion constant (J) for hollow rectangular sections can have errors >0.04 consider using J_sp for more accurate results", UserWarning)
        return self._J
    
    @J.setter
    def J(self, value):
        self._J = value
    
    @property
    def kx(self):
        raise NotImplementedError("Shear coefficient kx is not implemented for hollow rectangular sections, use kx_sp instead")
    
    @kx.setter
    def kx(self, value):
        self._kx = None
    
    @property 
    def ky(self):
        raise NotImplementedError("Shear coefficient ky is not implemented for hollow rectangular sections, use kx_sp instead")
    
    @ky.setter
    def ky(self, value):
        self._ky = None

class naca4_sym_airfoil(Shape):
    def __init__(self, naca='0012', chord=1, aero_center = 0.25, n_points=200, samplingStyle='cosine', plot=False, mesh_size=0, v=0.33):
        super().__init__()
        # Generates coordinates for a symmetric NACA 00xx airfoil.
        # Args:
        #     naca: NACA 4-digit airfoil code, limiting to symmetric airfoils for now
        #     chord: Chord length
        #     aero_center: location of the aerodynamic center as a fraction of the chord
        #     n_points: Number of points on the airfoil
        #     samplingStyle: 'linear' or 'cosine' or 'halfCosine' for the distribution of points along the chord
        match samplingStyle:
            case 'linear':
                self.x = np.linspace(0, chord, n_points+1) 
            case 'cosine':
                self.x = 0.5  * (1 - np.cos(np.linspace(0, np.pi, n_points+1))) 
            case 'halfCosine':
                self.x = 0.5  * (1 - np.cos(np.linspace(0, np.pi/2, n_points+1)))
            case _:
                raise ValueError('Invalid samplingStyle, choose from "linear", "cosine", "halfCosine"')

        self.t = int(naca[2:])/100
        self.y = 5 * self.t  * (0.2969 * np.sqrt(self.x) - 0.1260 * (self.x) - 
                                       0.3516 * (self.x)**2 + 0.2843 * (self.x)**3 - 0.1036 * (self.x)**4) # last coeff 0.1015 & 0.1036
        self.y[-1] = 0.0 # to avoid -0.0
        self.x = (self.x - aero_center) * chord
        self.y = self.y * chord

        # Rolling up the reversed coordinates to the front, remove the last point to avoid duplication
        # reverse X so that we start from the trailing edge and end at the trailing edge
        self.x = np.concatenate([self.x[::-1][:-1], self.x])
        self.y = np.concatenate([-self.y[::-1][:-1], self.y])

        self.kx = None
        self.ky = None

        # Origin info
        self.origin = (0, 0)
        self.sp2origin =  (0, 0)


        if plot:
            plt.plot(self.x, self.y, 'b')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'NACA {naca} Airfoil')
            plt.axis('equal')

        if sp_lib:
            # creating properties for the airfoil compunded geometry, formated as a list of tuples
            points = [(self.x[i], self.y[i]) for i in range(len(self.x))]
            # connectivity of the points, including the last point back to the first point for a closed geometry
            facets = [(i, i+1) for i in range(len(self.x)-1)] + [(len(self.x)-1, 0)]


            self.geom = CompoundGeometry.from_points(
                points=points,
                facets=facets,
                control_points=[(0,0)]
            )
            self.geom.create_mesh(mesh_sizes=[0.1])
            self.section = sectionProp_SectionAnalysis(geometry=self.geom, plot=plot, mesh_size=mesh_size)

            # setting Ixx & Iyy to geometric axis for symmetry, therfore being about BeamDyn 0-origin
            self.area_sp, _, _, self.Ixx_sp, self.Iyy_sp, self.J_sp, self.kx_sp, self.ky_sp = extractSecProps(self.section, getShearCoeff=True, v=v)   



        else:
            Warning('sectionproperties not installed, airfoil shapes will not be available')
            pass


# pytest
