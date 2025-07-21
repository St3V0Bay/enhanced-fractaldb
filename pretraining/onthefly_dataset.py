import torch
import torchvision
from torch.utils.data import Dataset
import sys
import os
sys.path.append(os.path.abspath('.'))  # Add current directory to path
from fractal_renderer.mathematical_functions import MathematicalFunctions
import numpy as np
from PIL import Image
import random

class OnTheFlyMathematicalFunctionDataset(Dataset):
    """
    PyTorch Dataset that generates rendered mathematical function images on the fly
    for training, ensuring efficient data feeding without I/O bottlenecks.
    Features comprehensive randomization for enhanced data augmentation.
    """
    def __init__(self, num_samples, image_size=224, transform=None):
        self.num_samples = num_samples
        self.image_size = image_size
        self.transform = transform
        self.configs = self._create_configs()

    def _create_configs(self):
        return [
            # Include all function types available
            "spherical_harmonics",
            "hyperbolic_functions", 
            "fractal_functions",
            "n_dimensional_curves",
            "complex_manifolds"
        ]

    def _random_params(self, function_type):
        """Generate randomized parameters for each mathematical function type"""
        if function_type == "spherical_harmonics":
            return {
                'l_max': random.randint(2, 10),  # Broader range for more diversity
                'm_range': random.choice([None, range(-2, 3), range(-1, 2)]),  # Sometimes limit m range
                'resolution': random.randint(30, 200),  # Variable resolution for regularization
                'radius': random.uniform(0.5, 2.0)  # Wider radius range
            }
        elif function_type == "hyperbolic_functions":
            return {
                'function_type': random.choice(['sinh', 'cosh', 'tanh', 'hyperboloid']),
                'u_range': (random.uniform(-3, -1), random.uniform(1, 3)),  # Random parameter ranges
                'v_range': (random.uniform(-3, -1), random.uniform(1, 3)),
                'resolution': random.randint(40, 200),
                'scale': random.uniform(0.3, 2.0)
            }
        elif function_type == "fractal_functions":
            fractal_type = random.choice(['mandelbrot', 'julia', 'burning_ship', 'newton'])
            base_params = {'iterations': random.randint(30, 300)}
            
            # Add type-specific randomized parameters
            if fractal_type == 'mandelbrot':
                base_params.update({
                    'center': (random.uniform(-2, 2), random.uniform(-2, 2)),
                    'zoom': random.uniform(0.5, 5.0)
                })
            elif fractal_type == 'julia':
                base_params.update({
                    'c_real': random.uniform(-1.5, 0.5),
                    'c_imag': random.uniform(-1.5, 1.5)
                })
            elif fractal_type == 'burning_ship':
                base_params.update({
                    'zoom': random.uniform(0.3, 3.0)
                })
            elif fractal_type == 'newton':
                base_params.update({
                    'power': random.randint(2, 6)
                })
            
            return {
                'fractal_type': fractal_type,
                'params': base_params,
                'scale': random.uniform(0.3, 2.0)
            }
        elif function_type == "n_dimensional_curves":
            curve_type = random.choice(['lissajous', 'rose', 'spiral', 'torus_knot'])
            
            # Generate randomized curve-specific parameters
            if curve_type == 'lissajous':
                params = {
                    'a': random.randint(1, 7),
                    'b': random.randint(1, 7), 
                    'A': random.uniform(0.5, 2.0),
                    'B': random.uniform(0.5, 2.0),
                    'delta': random.uniform(0, 2*np.pi),
                    'c': random.randint(1, 8)
                }
            elif curve_type == 'rose':
                params = {'k': random.uniform(1.5, 8.5)}  # Can be fractional
            elif curve_type == 'spiral':
                params = {'pitch': random.uniform(0.05, 0.3)}
            elif curve_type == 'torus_knot':
                params = {
                    'p': random.randint(2, 8),
                    'q': random.randint(3, 9), 
                    'R': random.uniform(1.5, 3.0),
                    'r': random.uniform(0.3, 1.5)
                }
            else:
                params = None
            
            return {
                'curve_type': curve_type,
                'dimensions': 3,
                'resolution': random.randint(200, 2000),  # Higher resolution range
                'params': params,
                'scale': random.uniform(0.4, 1.8)
            }
        elif function_type == "complex_manifolds":
            return {
                'manifold_type': random.choice(['klein_bottle', 'mobius_strip', 'boy_surface']),
                'resolution': random.randint(25, 80),  # Variable resolution for regularization
                'scale': random.uniform(0.3, 2.5)
            }

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        function_type = self.configs[idx % len(self.configs)]
        params = self._random_params(function_type)

        math_func = MathematicalFunctions(save_root='', function_name='', function_weight_count='00')

        if function_type == "spherical_harmonics":
            math_func.spherical_harmonics(**params)
        elif function_type == "hyperbolic_functions":
            math_func.hyperbolic_functions(**params)
        elif function_type == "fractal_functions":
            math_func.fractal_functions(**params)
        elif function_type == "n_dimensional_curves":
            math_func.n_dimensional_curves(**params)
        elif function_type == "complex_manifolds":
            math_func.complex_manifolds(**params)

        img_np = math_func.render_image(
            image_x=self.image_size,
            image_y=self.image_size,
            pad_x=0,
            pad_y=0,
            projection_type=random.choice(["orthographic", "perspective"]),  # Randomize projection
            view_angle=(random.uniform(0, np.pi), random.uniform(0, 2*np.pi)),
            count=0,
            render_type=random.choice(["point", "patch"]),  # Randomize render type
            white_background=True,
            return_img=True  # Important: return image array instead of saving
        )
        
        if img_np is None:
            # Fallback: create a blank image if generation fails
            img_np = np.full((self.image_size, self.image_size, 3), fill_value=255, dtype=np.uint8)

        img = Image.fromarray(img_np)

        if self.transform:
            img = self.transform(img)
        else:
            img = torchvision.transforms.ToTensor()(img)

        label = idx % len(self.configs)

        return img, label

