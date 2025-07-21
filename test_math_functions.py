#!/usr/bin/env python3
"""
Simple test for mathematical functions without torchvision dependencies
"""

import sys
import os
import numpy as np
from PIL import Image

from fractal_renderer.mathematical_functions import MathematicalFunctions

def test_mathematical_functions():
    """Test each mathematical function type"""
    print("Testing MathematicalFunctions...")
    
    # Test spherical harmonics
    print("\n1. Testing spherical harmonics...")
    math_func = MathematicalFunctions('', '', '')
    math_func.spherical_harmonics(l_max=3, resolution=50)
    
    img = math_func.render_image(224, 224, 0, 0, return_img=True)
    if img is not None:
        print(f"   Spherical harmonics: Generated image shape {img.shape}")
    else:
        print("   Spherical harmonics: Failed to generate image")
    
    # Test hyperbolic functions
    print("\n2. Testing hyperbolic functions...")
    math_func = MathematicalFunctions('', '', '')
    math_func.hyperbolic_functions(function_type="sinh", resolution=50)
    
    img = math_func.render_image(224, 224, 0, 0, return_img=True)
    if img is not None:
        print(f"   Hyperbolic functions: Generated image shape {img.shape}")
    else:
        print("   Hyperbolic functions: Failed to generate image")
    
    # Test fractal functions
    print("\n3. Testing fractal functions...")
    math_func = MathematicalFunctions('', '', '')
    math_func.fractal_functions(fractal_type="mandelbrot", params={'iterations': 50})
    
    img = math_func.render_image(224, 224, 0, 0, return_img=True)
    if img is not None:
        print(f"   Fractal functions: Generated image shape {img.shape}")
    else:
        print("   Fractal functions: Failed to generate image")
    
    # Test n-dimensional curves
    print("\n4. Testing n-dimensional curves...")
    math_func = MathematicalFunctions('', '', '')
    math_func.n_dimensional_curves(curve_type="lissajous", resolution=500)
    
    img = math_func.render_image(224, 224, 0, 0, return_img=True)
    if img is not None:
        print(f"   N-dimensional curves: Generated image shape {img.shape}")
    else:
        print("   N-dimensional curves: Failed to generate image")
    
    # Test complex manifolds
    print("\n5. Testing complex manifolds...")
    math_func = MathematicalFunctions('', '', '')
    math_func.complex_manifolds(manifold_type="klein_bottle", resolution=40)
    
    img = math_func.render_image(224, 224, 0, 0, return_img=True)
    if img is not None:
        print(f"   Complex manifolds: Generated image shape {img.shape}")
    else:
        print("   Complex manifolds: Failed to generate image")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    test_mathematical_functions()
