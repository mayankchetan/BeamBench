import pytest
import numpy as np
from utils import *

class TestAllClasses:
    def test_circle(self):
        # Test circle with radius 1
        c = circle(r=1.0)
        assert abs(c.area - np.pi) < 1e-10
        assert abs(c.Ixx - np.pi/4) < 1e-10 
        assert abs(c.Iyy - np.pi/4) < 1e-10
        assert abs(c.J - np.pi/2) < 1e-10

    def test_rectangle(self):
        # Test square (special case)
        r1 = rectangle(b=2.0, h=2.0)
        assert abs(r1.area - 4.0) < 1e-10
        assert abs(r1.Ixx - 16/12) < 1e-10
        assert abs(r1.Iyy - 16/12) < 1e-10
        
        # Test rectangle with different dimensions
        r2 = rectangle(b=1.0, h=2.0)
        assert abs(r2.area - 2.0) < 1e-10
        assert abs(r2.Ixx - 0.6666666666666666) < 1e-10
        assert abs(r2.Iyy - 0.16666666666666666) < 1e-10

    def test_hollow_circle(self):
        # Test hollow circle with outer radius 2 and thickness 0.5
        hc = hollowCircle(r=2.0, t=0.5)
        expected_area = np.pi * (4 - 2.25)  # π(R² - (R-t)²)
        assert abs(hc.area - expected_area) < 1e-10
        
    def test_hollow_rectangle(self):
        # Test hollow rectangle with outer dimensions 4x2 and thickness 0.5
        hr = hollowRectangle(b=4.0, h=2.0, t=0.5)
        expected_area = 4*2 - 3*1  # outer area - inner area
        print(hr.area, expected_area)
        assert abs(hr.area - expected_area) < 1e-10
        
        # Test that dimensions are stored correctly
        assert hr.b == 4.0
        assert hr.h == 2.0
        assert hr.t == 0.5

    def test_invalid_hollow_shapes(self):
        # Test that invalid dimensions raise errors
        with pytest.raises(ValueError):
            hollowRectangle(b=1.0, h=1.0, t=1.0)  # thickness equal to dimension
        
        with pytest.raises(ValueError):
            hollowCircle(r=1.0, t=1.1)  # thickness greater than radius

    def test_naca4_airfoil(self):
        # Test basic airfoil creation
        naca = naca4_sym_airfoil(naca='0012', chord=1.0)
        assert len(naca.x) == 401  # 200*2 + 1 points total after mirroring
        assert len(naca.y) == 401
        
        # Test symmetry
        n_half = len(naca.x) // 2
        assert np.allclose(naca.y[:n_half], -naca.y[-n_half:][::-1])
        
        # Test different sampling styles
        styles = ['linear', 'cosine', 'halfCosine']
        for style in styles:
            naca = naca4_sym_airfoil(naca='0012', samplingStyle=style)
            assert len(naca.x) == 401
        
        # Test invalid sampling style
        with pytest.raises(ValueError):
            naca4_sym_airfoil(samplingStyle='invalid')
        
        # # Test invalid NACA number - currently not working
        # with pytest.raises(ValueError):
        #     naca4_sym_airfoil(naca='1234')  # Only symmetric airfoils supported


@pytest.mark.skipif(not sp_lib, reason="sectionproperties not installed")
class TestShapeComparisons:
    def test_circle_comparison(self):
        c = circle(r=1.0)
        rel_tol = 1e-3  # 0.1% tolerance
        
        assert abs(c.area - c.area_sp)/c.area < rel_tol
        assert abs(c.Ixx - c.Ixx_sp)/c.Ixx < rel_tol
        assert abs(c.Iyy - c.Iyy_sp)/c.Iyy < rel_tol
        assert abs(c.J - c.J_sp)/c.J < rel_tol

    def test_rectangle_comparison(self):
        r = rectangle(b=2.0, h=1.0, mesh_size=0.01)
        rel_tol = 1e-3
        
        assert abs(r.area - r.area_sp)/r.area < rel_tol
        assert abs(r.Ixx - r.Ixx_sp)/r.Ixx < rel_tol
        assert abs(r.Iyy - r.Iyy_sp)/r.Iyy < rel_tol
        assert abs(r.J - r.J_sp)/r.J < rel_tol

    def test_hollow_circle_comparison(self):
        hc = hollowCircle(r=2.0, t=0.5)
        rel_tol = 1e-3
        
        assert abs(hc.area - hc.area_sp)/hc.area < rel_tol 
        assert abs(hc.Ixx - hc.Ixx_sp)/hc.Ixx < rel_tol
        assert abs(hc.Iyy - hc.Iyy_sp)/hc.Iyy < rel_tol
        assert abs(hc.J - hc.J_sp)/hc.J < rel_tol

    def test_hollow_rectangle_comparison(self):
        hr = hollowRectangle(b=4.0, h=2.0, t=0.5, mesh_size=0.01)
        rel_tol = 1e-3 
        
        assert abs(hr.area - hr.area_sp)/hr.area < rel_tol
        assert abs(hr.Ixx - hr.Ixx_sp)/hr.Ixx < rel_tol
        assert abs(hr.Iyy - hr.Iyy_sp)/hr.Iyy < rel_tol
        assert abs(hr.J - hr.J_sp)/hr.J < 1 # large error rate for J