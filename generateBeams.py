# Generating beams for the beam benchmark

import numpy as np
import matplotlib.pyplot as plt
from utils import *
import os
import welib.fast.beamdyn as bd

saveFolder = 'initialBeams'

# Beam properties
length = 10 #m
nodes = 21
nModes = 12
span = np.linspace(0, 1, nodes)

# Material properties
E = 210e9 #Pa
G = 80e9  #Pa
rho = 7850 #kg/m^3
v = 0.3

# BeamDyn damping
mu = np.ones(6) * 0.01


## Circular cross-section
r = 0.1
c = circle(r=r, mesh_size=0.01, v=v)
# BeamDyn matrices
KK, MM = c.getBeamDynMats(E=E, G=G, Nu=v, rho=rho)
# repeat the matrix KK & MM len(span) times
lKK = np.tile(KK, (len(span), 1, 1))
lMM = np.tile(MM, (len(span), 1, 1))
bd.write_beamdyn_sections(f'{saveFolder}/circularSection.dat', span, lKK, lMM, mu, Label=f'Circular section with radius {r}')

## Square cross-section
b = 0.1
h = 0.1
sq = rectangle(b=b, h=h, mesh_size=0.01, v=v)
# BeamDyn matrices
KK, MM = sq.getBeamDynMats(E=E, G=G, Nu=v, rho=rho)
# repeat the matrix KK & MM len(span) times
lKK = np.tile(KK, (len(span), 1, 1))
lMM = np.tile(MM, (len(span), 1, 1))
bd.write_beamdyn_sections(f'{saveFolder}/squareSection.dat', span, lKK, lMM, mu, Label=f'Square section with side b={b}, h={h}')

## Rectangular cross-section
b = 0.2
h = 0.1
r = rectangle(b=b, h=h, mesh_size=0.01, v=v)
# BeamDyn matrices
KK, MM = r.getBeamDynMats(E=E, G=G, Nu=v, rho=rho)
# repeat the matrix KK & MM len(span) times
lKK = np.tile(KK, (len(span), 1, 1))
lMM = np.tile(MM, (len(span), 1, 1))
bd.write_beamdyn_sections(f'{saveFolder}/rectangularSection.dat', span, lKK, lMM, mu, Label=f'Rectangular section with side b={b}, h={h}')

## Hollow circular cross-section
r = 0.2
t = 0.05
hc = hollowCircle(r=r, t=t, mesh_size=0.01, v=v)
# BeamDyn matrices
KK, MM = hc.getBeamDynMats(E=E, G=G, Nu=v, rho=rho)
# repeat the matrix KK & MM len(span) times
lKK = np.tile(KK, (len(span), 1, 1))
lMM = np.tile(MM, (len(span), 1, 1))
bd.write_beamdyn_sections(f'{saveFolder}/hollowCircularSection.dat', span, lKK, lMM, mu, Label=f'Hollow circular section with outer radius {r}, thickness {t}')

## Hollow rectangular cross-section
b = 0.2
h = 0.1
t = 0.02
hr = hollowRectangle(b=b, h=h, t=t, mesh_size=0.01, v=v)
# BeamDyn matrices
KK, MM = hr.getBeamDynMats(E=E, G=G, Nu=v, rho=rho)
# repeat the matrix KK & MM len(span) times
lKK = np.tile(KK, (len(span), 1, 1))
lMM = np.tile(MM, (len(span), 1, 1))
bd.write_beamdyn_sections(f'{saveFolder}/hollowRectangularSection.dat', span, lKK, lMM, mu, Label=f'Hollow rectangular section with outer side b={b}, h={h}, thickness {t}')


## NACA 0012 cross-section
chord = 0.1
naca0012 = naca4_sym_airfoil('0012', chord=0.1, mesh_size=0.01, v=v)
# BeamDyn matrices
KK, MM = naca0012.getBeamDynMats(E=E, G=G, Nu=v, rho=rho)
# repeat the matrix KK & MM len(span) times
lKK = np.tile(KK, (len(span), 1, 1))
lMM = np.tile(MM, (len(span), 1, 1))
bd.write_beamdyn_sections(f'{saveFolder}/naca0012Section.dat', span, lKK, lMM, mu, Label=f'NACA 0012 airfoil with chord {chord}')


## NACA 0020 cross-section
chord = 0.1
naca0020 = naca4_sym_airfoil('0020', chord=0.1, mesh_size=0.01, v=v)
# BeamDyn matrices
KK, MM = naca0020.getBeamDynMats(E=E, G=G, Nu=v, rho=rho)
# repeat the matrix KK & MM len(span) times
lKK = np.tile(KK, (len(span), 1, 1))
lMM = np.tile(MM, (len(span), 1, 1))
bd.write_beamdyn_sections(f'{saveFolder}/naca0020Section.dat', span, lKK, lMM, mu, Label=f'NACA 0020 airfoil with chord {chord}')



















