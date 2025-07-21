# -*- coding: utf-8 -*-
"""
Enhanced FractalDB Constructor with Higher-Dimensional Mathematical Functions
Supports fractal geometries, spherical harmonics, hyperbolic functions, and n-dimensional curves

@author: Enhanced FractalDB Implementation
"""

import os
import time
import argparse
import numpy as np
import random

from ifs_function import ifs_function
from mathematical_functions import MathematicalFunctions

def conf():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_root', default='./data', type=str, help='load csv root')
    parser.add_argument('--save_root', default='./data/EnhancedFractalDB', type=str, help='save png root')
    parser.add_argument('--image_size_x', default=512, type=int, help='image size x')
    parser.add_argument('--image_size_y', default=512, type=int, help='image size y')
    parser.add_argument('--pad_size_x', default=6, type=int, help='padding size x')
    parser.add_argument('--pad_size_y', default=6, type=int, help='padding size y')
    parser.add_argument('--iteration', default=100000, type=int, help='iteration for fractals')
    parser.add_argument('--draw_type', default='patch_gray', type=str, 
                       help='{point, patch}_{gray, color}')
    parser.add_argument('--weight_csv', default='./fractal_renderer/weights/weights_0.1.csv', 
                       type=str, help='weight parameter for fractals')
    parser.add_argument('--instance', default=10, type=int, 
                       help='#instance, 10 => 1000 instance, 100 => 10,000 instance per category')
    
    # Enhanced mathematical function parameters
    parser.add_argument('--enable_math_functions', default=True, type=bool, 
                       help='Enable mathematical functions generation')
    parser.add_argument('--math_categories', default=1000, type=int, 
                       help='Number of mathematical function categories')
    parser.add_argument('--spherical_harmonics_ratio', default=0.3, type=float,
                       help='Ratio of spherical harmonics in dataset')
    parser.add_argument('--hyperbolic_functions_ratio', default=0.3, type=float,
                       help='Ratio of hyperbolic functions in dataset')
    parser.add_argument('--parametric_curves_ratio', default=0.2, type=float,
                       help='Ratio of parametric curves in dataset')
    parser.add_argument('--complex_manifolds_ratio', default=0.2, type=float,
                       help='Ratio of complex manifolds in dataset')
    parser.add_argument('--white_background', action='store_true', 
                       help='Use white background for images instead of black')
    
    args = parser.parse_args()
    return args

def make_directory(save_root, name):
    if not os.path.exists(os.path.join(save_root, name)):
        os.makedirs(os.path.join(save_root, name), exist_ok=True)

def generate_mathematical_function_parameters():
    """Generate random parameters for mathematical functions"""
    
    # Spherical harmonics parameters
    spherical_params = {
        'l_max': random.randint(2, 6),
        'm_range': None,  # Use full range
        'resolution': random.randint(50, 150),
        'radius': random.uniform(0.5, 2.0)
    }
    
    # Hyperbolic function parameters
    hyperbolic_types = ["sinh", "cosh", "tanh", "hyperboloid"]
    hyperbolic_params = {
        'function_type': random.choice(hyperbolic_types),
        'u_range': (random.uniform(-3, -1), random.uniform(1, 3)),
        'v_range': (random.uniform(-3, -1), random.uniform(1, 3)),
        'resolution': random.randint(50, 150),
        'scale': random.uniform(0.5, 2.0)
    }
    
    # Parametric curve parameters
    curve_types = ["lissajous", "rose", "spiral", "torus_knot"]
    curve_type = random.choice(curve_types)
    
    if curve_type == "lissajous":
        curve_params = {
            'curve_type': curve_type,
            'dimensions': random.choice([2, 3]),
            'resolution': random.randint(500, 1500),
            'params': {
                'a': random.randint(1, 8),
                'b': random.randint(1, 8),
                'c': random.randint(1, 8),
                'A': random.uniform(0.5, 2.0),
                'B': random.uniform(0.5, 2.0),
                'delta': random.uniform(0, 2*np.pi)
            },
            'scale': random.uniform(0.5, 2.0)
        }
    elif curve_type == "rose":
        curve_params = {
            'curve_type': curve_type,
            'dimensions': random.choice([2, 3]),
            'resolution': random.randint(500, 1500),
            'params': {'k': random.uniform(1.5, 8.5)},
            'scale': random.uniform(0.5, 2.0)
        }
    elif curve_type == "spiral":
        curve_params = {
            'curve_type': curve_type,
            'dimensions': 3,
            'resolution': random.randint(500, 1500),
            'params': {'pitch': random.uniform(0.05, 0.3)},
            'scale': random.uniform(0.5, 2.0)
        }
    else:  # torus_knot
        curve_params = {
            'curve_type': curve_type,
            'dimensions': 3,
            'resolution': random.randint(500, 1500),
            'params': {
                'p': random.randint(2, 5),
                'q': random.randint(3, 7),
                'R': random.uniform(1.5, 3.0),
                'r': random.uniform(0.5, 1.5)
            },
            'scale': random.uniform(0.5, 2.0)
        }
    
    # Complex manifold parameters
    manifold_types = ["klein_bottle", "mobius_strip", "boy_surface"]
    manifold_params = {
        'manifold_type': random.choice(manifold_types),
        'resolution': random.randint(30, 80),
        'scale': random.uniform(0.5, 2.0)
    }
    
    return {
        'spherical': spherical_params,
        'hyperbolic': hyperbolic_params,
        'curves': curve_params,
        'manifolds': manifold_params
    }

def generate_mathematical_categories(args):
    """Generate mathematical function categories"""
    categories = []
    total_math_categories = args.math_categories
    
    # Calculate number of categories for each type
    num_spherical = int(total_math_categories * args.spherical_harmonics_ratio)
    num_hyperbolic = int(total_math_categories * args.hyperbolic_functions_ratio)
    num_curves = int(total_math_categories * args.parametric_curves_ratio)
    num_manifolds = total_math_categories - num_spherical - num_hyperbolic - num_curves
    
    # Generate spherical harmonics categories
    for i in range(num_spherical):
        params = generate_mathematical_function_parameters()
        categories.append({
            'type': 'spherical_harmonics',
            'name': f'spherical_{i:05d}',
            'params': params['spherical']
        })
    
    # Generate hyperbolic function categories
    for i in range(num_hyperbolic):
        params = generate_mathematical_function_parameters()
        categories.append({
            'type': 'hyperbolic_functions',
            'name': f'hyperbolic_{i:05d}',
            'params': params['hyperbolic']
        })
    
    # Generate parametric curve categories
    for i in range(num_curves):
        params = generate_mathematical_function_parameters()
        categories.append({
            'type': 'n_dimensional_curves',
            'name': f'curves_{i:05d}',
            'params': params['curves']
        })
    
    # Generate complex manifold categories
    for i in range(num_manifolds):
        params = generate_mathematical_function_parameters()
        categories.append({
            'type': 'complex_manifolds',
            'name': f'manifolds_{i:05d}',
            'params': params['manifolds']
        })
    
    return categories

if __name__ == "__main__":
    starttime = time.time()
    args = conf()
    
    # Create save directory
    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root, exist_ok=True)
    
    # Process original fractal categories (if CSV files exist)
    if os.path.exists(args.load_root):
        csv_names = [f for f in os.listdir(args.load_root) if f.endswith('.csv')]
        csv_names.sort()
        
        # Load weights for fractal rendering
        weights = []
        if os.path.exists(args.weight_csv):
            weights = np.genfromtxt(args.weight_csv, dtype=np.str_, delimiter=',')
            if len(weights.shape) == 1:  # Handle single row
                weights = [weights]
        
        print(f"Processing {len(csv_names)} fractal categories...")
        
        for csv_name in csv_names:
            name, ext = os.path.splitext(csv_name)
            print(f"Processing fractal: {name}")
            
            make_directory(args.save_root, name)
            fractal_weight = 0
            
            for weight in weights:
                padded_fractal_weight = '%02d' % fractal_weight
                
                if args.draw_type == 'point_gray':
                    generators = ifs_function(prev_x=0.0, prev_y=0.0, save_root=args.save_root,
                                            fractal_name=name, fractal_weight_count=padded_fractal_weight)
                    params = np.genfromtxt(os.path.join(args.load_root, csv_name), 
                                         dtype=np.str_, delimiter=',')
                    for param in params:
                        generators.set_param(float(param[0]), float(param[1]), float(param[2]), 
                                           float(param[3]), float(param[4]), float(param[5]), 
                                           float(param[6]),
                                           weight_a=float(weight[0]), weight_b=float(weight[1]), 
                                           weight_c=float(weight[2]), weight_d=float(weight[3]), 
                                           weight_e=float(weight[4]), weight_f=float(weight[5]))
                    generators.calculate(args.iteration)
                    generators.draw_point(args.image_size_x, args.image_size_y, 
                                        args.pad_size_x, args.pad_size_y, 'gray', 0)
                
                elif args.draw_type == 'point_color':
                    generators = ifs_function(prev_x=0.0, prev_y=0.0, save_root=args.save_root,
                                            fractal_name=name, fractal_weight_count=padded_fractal_weight)
                    params = np.genfromtxt(os.path.join(args.load_root, csv_name), 
                                         dtype=np.str_, delimiter=',')
                    for param in params:
                        generators.set_param(float(param[0]), float(param[1]), float(param[2]), 
                                           float(param[3]), float(param[4]), float(param[5]), 
                                           float(param[6]),
                                           weight_a=float(weight[0]), weight_b=float(weight[1]), 
                                           weight_c=float(weight[2]), weight_d=float(weight[3]), 
                                           weight_e=float(weight[4]), weight_f=float(weight[5]))
                    generators.calculate(args.iteration)
                    generators.draw_point(args.image_size_x, args.image_size_y, 
                                        args.pad_size_x, args.pad_size_y, 'color', 0)
                
                elif args.draw_type == 'patch_gray':
                    for count in range(args.instance):
                        generators = ifs_function(prev_x=0.0, prev_y=0.0, save_root=args.save_root,
                                                fractal_name=name, fractal_weight_count=padded_fractal_weight)
                        params = np.genfromtxt(os.path.join(args.load_root, csv_name), 
                                             dtype=np.str_, delimiter=',')
                        for param in params:
                            generators.set_param(float(param[0]), float(param[1]), float(param[2]), 
                                               float(param[3]), float(param[4]), float(param[5]), 
                                               float(param[6]),
                                               weight_a=float(weight[0]), weight_b=float(weight[1]), 
                                               weight_c=float(weight[2]), weight_d=float(weight[3]), 
                                               weight_e=float(weight[4]), weight_f=float(weight[5]))
                        generators.calculate(args.iteration)
                        generators.draw_patch(args.image_size_x, args.image_size_y, 
                                            args.pad_size_x, args.pad_size_y, 'gray', count)
                
                elif args.draw_type == 'patch_color':
                    for count in range(args.instance):
                        generators = ifs_function(prev_x=0.0, prev_y=0.0, save_root=args.save_root,
                                                fractal_name=name, fractal_weight_count=padded_fractal_weight)
                        params = np.genfromtxt(os.path.join(args.load_root, csv_name), 
                                             dtype=np.str_, delimiter=',')
                        for param in params:
                            generators.set_param(float(param[0]), float(param[1]), float(param[2]), 
                                               float(param[3]), float(param[4]), float(param[5]), 
                                               float(param[6]),
                                               weight_a=float(weight[0]), weight_b=float(weight[1]), 
                                               weight_c=float(weight[2]), weight_d=float(weight[3]), 
                                               weight_e=float(weight[4]), weight_f=float(weight[5]))
                        generators.calculate(args.iteration)
                        generators.draw_patch(args.image_size_x, args.image_size_y, 
                                            args.pad_size_x, args.pad_size_y, 'color', count)
                fractal_weight += 1
    
    # Process mathematical function categories
    if args.enable_math_functions:
        print(f"\\nGenerating {args.math_categories} mathematical function categories...")
        math_categories = generate_mathematical_categories(args)
        
        for category in math_categories:
            print(f"Processing {category['type']}: {category['name']}")
            make_directory(args.save_root, category['name'])
            
            # Generate different variations (similar to fractal weights)
            for variation in range(min(5, len(weights) if len(weights) > 0 else 5)):  # Limit variations
                padded_variation = '%02d' % variation
                
                for count in range(args.instance):
                    # Create mathematical function generator
                    math_gen = MathematicalFunctions(
                        save_root=args.save_root,
                        function_name=category['name'],
                        function_weight_count=padded_variation
                    )
                    
                    # Add some randomization for each instance
                    params = category['params'].copy()
                    
                    # Generate the mathematical function
                    if category['type'] == 'spherical_harmonics':
                        # Add slight parameter variation
                        params['radius'] *= random.uniform(0.8, 1.2)
                        params['l_max'] = max(1, params['l_max'] + random.randint(-1, 1))
                        math_gen.spherical_harmonics(**params)
                    
                    elif category['type'] == 'hyperbolic_functions':
                        # Add slight parameter variation
                        params['scale'] *= random.uniform(0.8, 1.2)
                        math_gen.hyperbolic_functions(**params)
                    
                    elif category['type'] == 'n_dimensional_curves':
                        # Add slight parameter variation
                        params['scale'] *= random.uniform(0.8, 1.2)
                        if 'params' in params and params['params']:
                            for key in params['params']:
                                if isinstance(params['params'][key], (int, float)):
                                    params['params'][key] *= random.uniform(0.9, 1.1)
                        math_gen.n_dimensional_curves(**params)
                    
                    elif category['type'] == 'complex_manifolds':
                        # Add slight parameter variation
                        params['scale'] *= random.uniform(0.8, 1.2)
                        math_gen.complex_manifolds(**params)
                    
                    # Render the image with random viewing angles
                    view_elevation = random.uniform(-np.pi/4, np.pi/4)
                    view_azimuth = random.uniform(0, 2*np.pi)
                    projection = random.choice(['orthographic', 'perspective'])
                    render_type = 'point' if 'point' in args.draw_type else 'patch'
                    
                    math_gen.render_image(
                        image_x=args.image_size_x,
                        image_y=args.image_size_y,
                        pad_x=args.pad_size_x,
                        pad_y=args.pad_size_y,
                        white_background=args.white_background,
                        projection_type=projection,
                        view_angle=(view_elevation, view_azimuth),
                        count=count,
                        render_type=render_type
                    )
    
    endtime = time.time()
    interval = endtime - starttime
    print(f"\\nCompleted enhanced FractalDB generation!")
    print(f"Time elapsed: {int(interval/3600)}h {int((interval%3600)/60)}m {int((interval%3600)%60)}s")
    print(f"Dataset saved to: {args.save_root}")
