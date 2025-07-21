#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Demonstration of Enhanced Mathematical Functions for Equation2Model
This script generates 10 diverse images for each class of mathematical functions:
- Spherical Harmonics (10 variations)
- Hyperbolic Functions (10 variations)
- N-Dimensional Curves (10 variations)
- Complex Manifolds (10 variations)

Each set shows maximum diversity in shape, form, and color.

@author: Equation2Model Implementation
"""

import os
import sys
import numpy as np
import random
import os
import numpy as np
from fractal_renderer.ifs_function import ifs_function
from fractal_renderer.mathematical_functions import MathematicalFunctions

def demo_spherical_harmonics():
    """Generate 10 diverse spherical harmonics examples"""
    print("Generating 10 diverse Spherical Harmonics examples...")

    demo_dir = "./demo_output"
    if not os.path.exists(demo_dir):
        os.makedirs(demo_dir)

    # Define 10 diverse parameter sets for spherical harmonics
    spherical_configs = [
        {"l_max": 2, "resolution": 60, "radius": 0.8, "view": (0.2, 0.5), "proj": "orthographic", "render": "point"},
        {"l_max": 4, "resolution": 80, "radius": 1.2, "view": (0.8, 1.2), "proj": "perspective", "render": "patch"},
        {"l_max": 6, "resolution": 100, "radius": 1.5, "view": (0.4, 2.1), "proj": "orthographic", "render": "point"},
        {"l_max": 3, "resolution": 120, "radius": 0.6, "view": (1.1, 0.8), "proj": "perspective", "render": "patch"},
        {"l_max": 8, "resolution": 90, "radius": 2.0, "view": (0.6, 3.0), "proj": "orthographic", "render": "point"},
        {"l_max": 5, "resolution": 140, "radius": 1.8, "view": (0.9, 1.7), "proj": "perspective", "render": "patch"},
        {"l_max": 7, "resolution": 70, "radius": 1.0, "view": (0.3, 2.5), "proj": "orthographic", "render": "point"},
        {"l_max": 9, "resolution": 110, "radius": 1.4, "view": (1.2, 0.4), "proj": "perspective", "render": "patch"},
        {"l_max": 4, "resolution": 160, "radius": 0.9, "view": (0.7, 2.8), "proj": "orthographic", "render": "point"},
        {"l_max": 6, "resolution": 85, "radius": 2.2, "view": (0.5, 1.9), "proj": "perspective", "render": "patch"},
    ]

    for i, config in enumerate(spherical_configs):
        print(f"  Creating spherical harmonics #{i+1}: l_max={config['l_max']}, resolution={config['resolution']}...")

        math_func = MathematicalFunctions(
            save_root=demo_dir,
            function_name=f"spherical_diverse_{i+1:02d}",
            function_weight_count="00"
        )

        # Generate with diverse parameters
        math_func.spherical_harmonics(
            l_max=config["l_max"],
            resolution=config["resolution"],
            radius=config["radius"]
        )

        # Render with varied settings
        math_func.render_image(
            image_x=512, image_y=512, pad_x=15, pad_y=15,
            projection_type=config["proj"],
            view_angle=config["view"],
            count=0,
            render_type=config["render"],
            white_background=True
        )

def demo_hyperbolic_functions():
    """Generate 10 diverse hyperbolic functions examples"""
    print("Generating 10 diverse Hyperbolic Functions examples...")

    demo_dir = "./demo_output"
    if not os.path.exists(demo_dir):
        os.makedirs(demo_dir)

    # Define 10 diverse hyperbolic function configurations
    hyperbolic_configs = [
        {"type": "sinh", "u_range": (-2, 2), "v_range": (-2, 2), "resolution": 80, "scale": 0.8, "view": (0.3, 0.7), "proj": "orthographic", "render": "point"},
        {"type": "cosh", "u_range": (-1.5, 1.5), "v_range": (-1.5, 1.5), "resolution": 100, "scale": 1.2, "view": (0.8, 1.5), "proj": "perspective", "render": "patch"},
        {"type": "tanh", "u_range": (-3, 3), "v_range": (-3, 3), "resolution": 120, "scale": 1.5, "view": (0.5, 2.1), "proj": "orthographic", "render": "point"},
        {"type": "hyperboloid", "u_range": (-2.5, 2.5), "v_range": (-np.pi, np.pi), "resolution": 90, "scale": 0.6, "view": (1.0, 0.4), "proj": "perspective", "render": "patch"},
        {"type": "sinh", "u_range": (-1, 1), "v_range": (-1, 1), "resolution": 140, "scale": 2.0, "view": (0.2, 2.8), "proj": "orthographic", "render": "point"},
        {"type": "cosh", "u_range": (-2.8, 2.8), "v_range": (-2.8, 2.8), "resolution": 70, "scale": 1.8, "view": (0.9, 1.1), "proj": "perspective", "render": "patch"},
        {"type": "tanh", "u_range": (-4, 4), "v_range": (-4, 4), "resolution": 110, "scale": 1.0, "view": (0.6, 2.5), "proj": "orthographic", "render": "point"},
        {"type": "hyperboloid", "u_range": (-1.8, 1.8), "v_range": (-2*np.pi, 2*np.pi), "resolution": 130, "scale": 1.4, "view": (1.2, 0.8), "proj": "perspective", "render": "patch"},
        {"type": "sinh", "u_range": (-2.2, 2.2), "v_range": (-2.2, 2.2), "resolution": 95, "scale": 0.9, "view": (0.4, 3.0), "proj": "orthographic", "render": "point"},
        {"type": "cosh", "u_range": (-1.2, 1.2), "v_range": (-1.2, 1.2), "resolution": 155, "scale": 2.2, "view": (0.7, 1.8), "proj": "perspective", "render": "patch"},
    ]

    for i, config in enumerate(hyperbolic_configs):
        print(f"  Creating hyperbolic function #{i+1}: {config['type']}, scale={config['scale']}...")

        math_func = MathematicalFunctions(
            save_root=demo_dir,
            function_name=f"hyperbolic_diverse_{i+1:02d}",
            function_weight_count="00"
        )

        # Generate with diverse parameters
        math_func.hyperbolic_functions(
            function_type=config["type"],
            u_range=config["u_range"],
            v_range=config["v_range"],
            resolution=config["resolution"],
            scale=config["scale"]
        )

        # Render with varied settings
        math_func.render_image(
            image_x=512, image_y=512, pad_x=15, pad_y=15,
            projection_type=config["proj"],
            view_angle=config["view"],
            count=0,
            render_type=config["render"],
            white_background=True
        )

def demo_parametric_curves():
    """Generate 10 diverse n-dimensional parametric curves"""
    print("Generating 10 diverse Parametric Curves examples...")

    demo_dir = "./demo_output"
    if not os.path.exists(demo_dir):
        os.makedirs(demo_dir)

    # Define 10 diverse parametric curve configurations
    curve_configs = [
        {"type": "lissajous", "params": {'a': 3, 'b': 5, 'c': 7}, "resolution": 1000, "scale": 0.8, "view": (0.6, 1.2), "proj": "orthographic", "render": "point"},
        {"type": "rose", "params": {'k': 6}, "resolution": 900, "scale": 1.2, "view": (0.8, 1.6), "proj": "perspective", "render": "patch"},
        {"type": "spiral", "params": {'pitch': 0.1}, "resolution": 1100, "scale": 1.5, "view": (0.7, 2.0), "proj": "orthographic", "render": "point"},
        {"type": "torus_knot", "params": {'p': 2, 'q': 5}, "resolution": 950, "scale": 1.0, "view": (1.1, 0.4), "proj": "perspective", "render": "patch"},
        {"type": "lissajous", "params": {'a': 5, 'b': 6, 'c': 8}, "resolution": 1200, "scale": 2.0, "view": (0.5, 3.1), "proj": "orthographic", "render": "point"},
        {"type": "rose", "params": {'k': 4.5}, "resolution": 800, "scale": 1.8, "view": (0.9, 1.3), "proj": "perspective", "render": "patch"},
        {"type": "spiral", "params": {'pitch': 0.2}, "resolution": 1300, "scale": 1.0, "view": (0.3, 2.6), "proj": "orthographic", "render": "point"},
        {"type": "torus_knot", "params": {'p': 3, 'q': 7}, "resolution": 1000, "scale": 1.3, "view": (1.4, 0.7), "proj": "perspective", "render": "patch"},
        {"type": "lissajous", "params": {'a': 7, 'b': 8, 'c': 9}, "resolution": 1150, "scale": 0.9, "view": (0.2, 0.9), "proj": "orthographic", "render": "point"},
        {"type": "rose", "params": {'k': 7}, "resolution": 1050, "scale": 2.3, "view": (0.3, 1.5), "proj": "perspective", "render": "patch"},
    ]

    for i, config in enumerate(curve_configs):
        print(f"  Creating parametric curve #{i+1}: {config['type']}, scale={config['scale']}...")

        math_func = MathematicalFunctions(
            save_root=demo_dir,
            function_name=f"curves_diverse_{i+1:02d}",
            function_weight_count="00"
        )

        # Generate with diverse parameters
        math_func.n_dimensional_curves(
            curve_type=config["type"],
            dimensions=3,
            resolution=config["resolution"],
            params=config["params"],
            scale=config["scale"]
        )

        # Render with varied settings
        math_func.render_image(
            image_x=512, image_y=512, pad_x=15, pad_y=15,
            projection_type=config["proj"],
            view_angle=config["view"],
            count=0,
            render_type=config["render"],
            white_background=True
        )

def demo_complex_manifolds():
    """Generate 10 diverse complex manifolds examples"""
    print("Generating 10 diverse Complex Manifolds examples...")

    demo_dir = "./demo_output"
    if not os.path.exists(demo_dir):
        os.makedirs(demo_dir)

    # Define 10 diverse complex manifold configurations
    manifold_configs = [
        {"type": "klein_bottle", "resolution": 50, "scale": 0.8, "view": (0.5, 1.0), "proj": "orthographic", "render": "point"},
        {"type": "mobius_strip", "resolution": 70, "scale": 1.2, "view": (0.6, 1.5), "proj": "perspective", "render": "patch"},
        {"type": "boy_surface", "resolution": 90, "scale": 1.5, "view": (0.4, 2.1), "proj": "orthographic", "render": "point"},
        {"type": "klein_bottle", "resolution": 85, "scale": 0.9, "view": (0.9, 0.8), "proj": "perspective", "render": "patch"},
        {"type": "mobius_strip", "resolution": 60, "scale": 2.0, "view": (0.3, 3.0), "proj": "orthographic", "render": "point"},
        {"type": "boy_surface", "resolution": 100, "scale": 1.8, "view": (1.0, 1.3), "proj": "perspective", "render": "patch"},
        {"type": "klein_bottle", "resolution": 75, "scale": 1.0, "view": (0.2, 2.5), "proj": "orthographic", "render": "point"},
        {"type": "mobius_strip", "resolution": 65, "scale": 1.4, "view": (1.2, 0.4), "proj": "perspective", "render": "patch"},
        {"type": "boy_surface", "resolution": 95, "scale": 0.7, "view": (0.7, 2.8), "proj": "orthographic", "render": "point"},
        {"type": "klein_bottle", "resolution": 80, "scale": 2.2, "view": (0.1, 1.9), "proj": "perspective", "render": "patch"},
    ]

    for i, config in enumerate(manifold_configs):
        print(f"  Creating complex manifold #{i+1}: {config['type']}, scale={config['scale']}...")

        math_func = MathematicalFunctions(
            save_root=demo_dir,
            function_name=f"manifold_diverse_{i+1:02d}",
            function_weight_count="00"
        )

        # Generate with diverse parameters
        math_func.complex_manifolds(
            manifold_type=config["type"],
            resolution=config["resolution"],
            scale=config["scale"]
        )

        # Render with varied settings
        math_func.render_image(
            image_x=512, image_y=512, pad_x=15, pad_y=15,
            projection_type=config["proj"],
            view_angle=config["view"],
            count=0,
            render_type=config["render"],
            white_background=True
        )

def demo_legacy_fractals():
    """Generate legacy fractal examples with proper rendering"""
    print("Generating Legacy Fractal examples...")

    legacy_demo_dir = "./demo_output/legacy"
    if not os.path.exists(legacy_demo_dir):
        os.makedirs(legacy_demo_dir)

    # Example legacy parameters
    params_list = [
        [0.85, 0.04, -0.04, 0.85, 0.0, 1.6],
        [0.2, -0.26, 0.23, 0.22, 0.0, 1.6],
        [-0.15, 0.28, 0.26, 0.24, 0.0, 0.44],
        [0.0, 0.0, 0.0, 0.16, 0.0, 0.0]
    ]

    weights = [0.85, 0.07, 0.07, 0.01]
    instance_count = 10

    for i in range(instance_count):
        fractional_name = f"legacy_example_{i+1:02d}"

        frac_dir = os.path.join(legacy_demo_dir, fractional_name)
        if not os.path.exists(frac_dir):
            os.makedirs(frac_dir)

        ifs_gen = ifs_function(
            prev_x=0.0, prev_y=0.0, save_root=legacy_demo_dir,
            fractal_name=fractional_name, fractal_weight_count="00"
        )

        for params in params_list:
            ifs_gen.set_param(*params, weight_a=weights[0], weight_b=weights[1],
                                weight_c=weights[2], weight_d=weights[3], proba=0.1)

        ifs_gen.calculate(iteration=50000)
        ifs_gen.draw_patch(image_x=512, image_y=512,
                           pad_x=15, pad_y=15, set_color='color', count=0)

    # Add proper fractal rendering following the same pattern as other functions
    demo_dir = "./demo_output"
    if not os.path.exists(demo_dir):
        os.makedirs(demo_dir)

    # Fractal configurations similar to other mathematical functions
    fractal_configs = [
        {"type": "mandelbrot", "params": {'iterations': 100, 'zoom': 1.0, 'center': (0, 0)}, "scale": 0.8, "view": (0.2, 0.5), "proj": "orthographic", "render": "point"},
        {"type": "julia", "params": {'c_real': -0.7, 'c_imag': 0.27, 'iterations': 80}, "scale": 1.2, "view": (0.8, 1.2), "proj": "perspective", "render": "patch"},
        {"type": "burning_ship", "params": {'iterations': 120, 'zoom': 1.5}, "scale": 1.5, "view": (0.4, 2.1), "proj": "orthographic", "render": "point"},
        {"type": "newton", "params": {'iterations': 60, 'power': 3}, "scale": 0.6, "view": (1.1, 0.8), "proj": "perspective", "render": "patch"},
        {"type": "mandelbrot", "params": {'iterations': 150, 'zoom': 2.0, 'center': (-0.5, 0.6)}, "scale": 2.0, "view": (0.6, 3.0), "proj": "orthographic", "render": "point"},
        {"type": "julia", "params": {'c_real': -0.8, 'c_imag': 0.156, 'iterations': 100}, "scale": 1.8, "view": (0.9, 1.7), "proj": "perspective", "render": "patch"},
        {"type": "burning_ship", "params": {'iterations': 90, 'zoom': 0.8}, "scale": 1.0, "view": (0.3, 2.5), "proj": "orthographic", "render": "point"},
        {"type": "newton", "params": {'iterations': 80, 'power': 4}, "scale": 1.4, "view": (1.2, 0.4), "proj": "perspective", "render": "patch"},
        {"type": "mandelbrot", "params": {'iterations': 200, 'zoom': 0.5, 'center': (-0.16, 1.04)}, "scale": 0.9, "view": (0.7, 2.8), "proj": "orthographic", "render": "point"},
        {"type": "julia", "params": {'c_real': -0.123, 'c_imag': 0.745, 'iterations': 90}, "scale": 2.2, "view": (0.5, 1.9), "proj": "perspective", "render": "patch"},
    ]

    for i, config in enumerate(fractal_configs):
        print(f"  Creating fractal #{i+1}: {config['type']}, scale={config['scale']}...")

        math_func = MathematicalFunctions(
            save_root=demo_dir,
            function_name=f"fractal_diverse_{i+1:02d}",
            function_weight_count="00"
        )

        # Generate with diverse parameters
        math_func.fractal_functions(
            fractal_type=config["type"],
            params=config["params"],
            scale=config["scale"]
        )

        # Render with varied settings
        math_func.render_image(
            image_x=512, image_y=512, pad_x=15, pad_y=15,
            projection_type=config["proj"],
            view_angle=config["view"],
            count=0,
            render_type=config["render"],
            white_background=True
        )


def create_comparison_summary():
    """Create a summary of the different function types"""
    print("\n=== Equation2Model Mathematical Functions Summary ===\n")

    print("1. SPHERICAL HARMONICS")
    print("   - Mathematical basis: Y_l^m(θ, φ) - solutions to Laplace equation on sphere")
    print("   - Applications: Quantum mechanics, computer graphics, geophysics")
    print("   - Parameters: degree (l), order (m), resolution, radius modulation")
    print("   - Visual characteristics: Complex spherical patterns with radial variations")
    print()

    print("2. HYPERBOLIC FUNCTIONS")
    print("   - Types: sinh, cosh, tanh surfaces, hyperboloids")
    print("   - Mathematical basis: Hyperbolic trigonometry, non-Euclidean geometry")
    print("   - Applications: Relativity, differential geometry, neural networks")
    print("   - Parameters: function type, parameter ranges, scaling")
    print("   - Visual characteristics: Saddle surfaces, exponential growth patterns")
    print()

    print("3. N-DIMENSIONAL PARAMETRIC CURVES")
    print("   - Types: Lissajous curves, rose curves, spirals, torus knots")
    print("   - Mathematical basis: Parametric equations in multiple dimensions")
    print("   - Applications: Oscillations, robotics, topology, knot theory")
    print("   - Parameters: frequency ratios, amplitudes, phases")
    print("   - Visual characteristics: Periodic, symmetric, knotted structures")
    print()

    print("4. COMPLEX MANIFOLDS")
    print("   - Types: Klein bottle, Möbius strip, Boy's surface")
    print("   - Mathematical basis: Differential topology, non-orientable surfaces")
    print("   - Applications: Topology, abstract algebra, theoretical physics")
    print("   - Parameters: resolution, scaling factors")
    print("   - Visual characteristics: Self-intersecting, twisted surfaces")
    print()

    print("=== Benefits for Neural Network Pre-training ===")
    print("• Increased geometric diversity beyond traditional fractals")
    print("• Rich mathematical structures with inherent symmetries")
    print("• Scalable complexity through parameter variation")
    print("• Natural data augmentation through projection angles")
    print("• Theoretical foundation for understanding learned representations")

if __name__ == "__main__":
    print("Equation2Model - Mathematical Functions Demonstration")
    print("=" * 55)

    try:
        # Create demo directory
        if not os.path.exists("demo_output"):
            os.makedirs("demo_output")

        # Run demonstrations
        demo_spherical_harmonics()
        print()
        demo_hyperbolic_functions()
        print()
        demo_parametric_curves()
        print()
        demo_complex_manifolds()
        print()

        demo_legacy_fractals()
        print()

        create_comparison_summary()

        print(f"\nDemo images saved to: {os.path.abspath('./demo_output')}")
        print("Each function type generates images with different transformations (flip0-3)")

    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required packages: numpy, scipy, PIL")
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
