# -*- coding: utf-8 -*-
"""
Higher-dimensional mathematical functions for Equation2Model
Implements spherical harmonics, hyperbolic functions, and n-dimensional curves

@author: Equation2Model Implementation
"""

import os
import math
import numpy as np
from PIL import Image
from scipy import special
import random

class MathematicalFunctions:
    def __init__(self, save_root, function_name, function_weight_count):
        self.save_root = save_root
        self.function_name = function_name
        self.function_weight_count = function_weight_count
        self.coordinates = {"x": [], "y": [], "z": []}
        self.colors = []
    
    def spherical_harmonics(self, l_max=5, m_range=None, resolution=100, radius=1.0):
        """
        Generate spherical harmonics Y_l^m(θ, φ)
        
        Args:
            l_max: Maximum degree of spherical harmonics
            m_range: Range of orders (-l ≤ m ≤ l), if None uses full range
            resolution: Resolution for theta and phi sampling
            radius: Radius of the sphere
        """
        self.coordinates = {"x": [], "y": [], "z": []}
        self.colors = []
        
        theta = np.linspace(0, np.pi, resolution)
        phi = np.linspace(0, 2*np.pi, resolution)
        theta_mesh, phi_mesh = np.meshgrid(theta, phi)
        
        for l in range(l_max + 1):
            m_list = m_range if m_range else range(-l, l + 1)
            for m in m_list:
                # Calculate spherical harmonics
                Y_lm = special.sph_harm(m, l, phi_mesh, theta_mesh)
                
                # Use magnitude for radius modulation
                r_modulated = radius * (1 + 0.3 * np.abs(Y_lm))
                
                # Convert to Cartesian coordinates
                x = r_modulated * np.sin(theta_mesh) * np.cos(phi_mesh)
                y = r_modulated * np.sin(theta_mesh) * np.sin(phi_mesh)
                z = r_modulated * np.cos(theta_mesh)
                
                # Flatten and store
                self.coordinates["x"].extend(x.flatten())
                self.coordinates["y"].extend(y.flatten())
                self.coordinates["z"].extend(z.flatten())
                
                # Color based on harmonic values
                colors = self._generate_harmonic_colors(Y_lm.flatten(), l, m)
                self.colors.extend(colors)
    
    def hyperbolic_functions(self, function_type="sinh", u_range=(-2, 2), v_range=(-2, 2), 
                           resolution=100, scale=1.0):
        """
        Generate hyperbolic surfaces and curves
        
        Args:
            function_type: Type of hyperbolic function ('sinh', 'cosh', 'tanh', 'hyperboloid')
            u_range, v_range: Parameter ranges
            resolution: Sampling resolution
            scale: Scaling factor
        """
        self.coordinates = {"x": [], "y": [], "z": []}
        self.colors = []
        
        u = np.linspace(u_range[0], u_range[1], resolution)
        v = np.linspace(v_range[0], v_range[1], resolution)
        u_mesh, v_mesh = np.meshgrid(u, v)
        
        if function_type == "sinh":
            # Sinh surface
            x = scale * np.sinh(u_mesh) * np.cos(v_mesh)
            y = scale * np.sinh(u_mesh) * np.sin(v_mesh)
            z = scale * np.cosh(u_mesh)
        elif function_type == "cosh":
            # Cosh surface
            x = scale * np.cosh(u_mesh) * np.cos(v_mesh)
            y = scale * np.cosh(u_mesh) * np.sin(v_mesh)
            z = scale * np.sinh(u_mesh)
        elif function_type == "tanh":
            # Tanh surface
            x = scale * u_mesh
            y = scale * v_mesh
            z = scale * np.tanh(np.sqrt(u_mesh**2 + v_mesh**2))
        elif function_type == "hyperboloid":
            # Hyperboloid of one sheet: x²/a² + y²/b² - z²/c² = 1
            a, b, c = scale, scale, scale
            x = a * np.cosh(u_mesh) * np.cos(v_mesh)
            y = b * np.cosh(u_mesh) * np.sin(v_mesh)
            z = c * np.sinh(u_mesh)
        
        # Store coordinates
        self.coordinates["x"].extend(x.flatten())
        self.coordinates["y"].extend(y.flatten())
        self.coordinates["z"].extend(z.flatten())
        
        # Generate colors based on function values
        colors = self._generate_hyperbolic_colors(x.flatten(), y.flatten(), z.flatten(), function_type)
        self.colors.extend(colors)
    
    def n_dimensional_curves(self, curve_type="lissajous", dimensions=3, resolution=1000, 
                           params=None, scale=1.0):
        """
        Generate n-dimensional parametric curves
        
        Args:
            curve_type: Type of curve ('lissajous', 'rose', 'spiral', 'torus_knot')
            dimensions: Number of dimensions (2 or 3 for visualization)
            resolution: Number of points
            params: Curve-specific parameters
            scale: Scaling factor
        """
        self.coordinates = {"x": [], "y": [], "z": []}
        self.colors = []
        
        t = np.linspace(0, 2*np.pi, resolution)
        
        if curve_type == "lissajous":
            # Lissajous curves: x = A*sin(at + δ), y = B*sin(bt)
            a, b = params.get('a', 3) if params else 3, params.get('b', 2) if params else 2
            A, B = params.get('A', 1) if params else 1, params.get('B', 1) if params else 1
            delta = params.get('delta', np.pi/2) if params else np.pi/2
            
            x = scale * A * np.sin(a * t + delta)
            y = scale * B * np.sin(b * t)
            if dimensions == 3:
                c = params.get('c', 4) if params else 4
                z = scale * np.sin(c * t)
            else:
                z = np.zeros_like(x)
        
        elif curve_type == "rose":
            # Rose curves: r = cos(k*θ)
            k = params.get('k', 3) if params else 3
            r = scale * np.cos(k * t)
            x = r * np.cos(t)
            y = r * np.sin(t)
            if dimensions == 3:
                z = scale * np.sin(2 * t)  # Add 3D component
            else:
                z = np.zeros_like(x)
        
        elif curve_type == "spiral":
            # 3D spiral
            pitch = params.get('pitch', 0.1) if params else 0.1
            x = scale * np.cos(t)
            y = scale * np.sin(t)
            z = scale * pitch * t
        
        elif curve_type == "torus_knot":
            # Torus knot: (p, q)-torus knot
            p, q = params.get('p', 2) if params else 2, params.get('q', 3) if params else 3
            R, r = params.get('R', 2) if params else 2, params.get('r', 1) if params else 1
            
            x = scale * (R + r * np.cos(q * t)) * np.cos(p * t)
            y = scale * (R + r * np.cos(q * t)) * np.sin(p * t)
            z = scale * r * np.sin(q * t)
        
        # Store coordinates
        self.coordinates["x"].extend(x)
        self.coordinates["y"].extend(y)
        self.coordinates["z"].extend(z)
        
        # Generate colors based on parameter t
        colors = self._generate_curve_colors(t, curve_type)
        self.colors.extend(colors)
    
    def complex_manifolds(self, manifold_type="klein_bottle", resolution=50, scale=1.0):
        """
        Generate complex mathematical manifolds
        
        Args:
            manifold_type: Type of manifold ('klein_bottle', 'mobius_strip', 'boy_surface')
            resolution: Sampling resolution
            scale: Scaling factor
        """
        self.coordinates = {"x": [], "y": [], "z": []}
        self.colors = []
        
        if manifold_type == "klein_bottle":
            u = np.linspace(0, 2*np.pi, resolution)
            v = np.linspace(0, 2*np.pi, resolution)
            u_mesh, v_mesh = np.meshgrid(u, v)
            
            # Klein bottle parametrization
            x = scale * (3 + np.cos(u_mesh/2)*np.sin(v_mesh) - np.sin(u_mesh/2)*np.sin(2*v_mesh)) * np.cos(u_mesh)
            y = scale * (3 + np.cos(u_mesh/2)*np.sin(v_mesh) - np.sin(u_mesh/2)*np.sin(2*v_mesh)) * np.sin(u_mesh)
            z = scale * (np.sin(u_mesh/2)*np.sin(v_mesh) + np.cos(u_mesh/2)*np.sin(2*v_mesh))
        
        elif manifold_type == "mobius_strip":
            u = np.linspace(0, 2*np.pi, resolution)
            v = np.linspace(-0.5, 0.5, resolution//4)
            u_mesh, v_mesh = np.meshgrid(u, v)
            
            # Möbius strip parametrization
            x = scale * (1 + v_mesh*np.cos(u_mesh/2)) * np.cos(u_mesh)
            y = scale * (1 + v_mesh*np.cos(u_mesh/2)) * np.sin(u_mesh)
            z = scale * v_mesh * np.sin(u_mesh/2)
        
        elif manifold_type == "boy_surface":
            u = np.linspace(0, np.pi, resolution)
            v = np.linspace(0, np.pi, resolution)
            u_mesh, v_mesh = np.meshgrid(u, v)
            
            # Boy's surface (simplified parametrization)
            x = scale * np.sin(u_mesh) * np.cos(v_mesh)
            y = scale * np.sin(u_mesh) * np.sin(v_mesh) 
            z = scale * np.cos(u_mesh) * np.sin(2*v_mesh)
        
        # Store coordinates
        self.coordinates["x"].extend(x.flatten())
        self.coordinates["y"].extend(y.flatten())
        self.coordinates["z"].extend(z.flatten())
        
        # Generate colors
        colors = self._generate_manifold_colors(len(x.flatten()), manifold_type)
        self.colors.extend(colors)
    
    def fractal_functions(self, fractal_type="mandelbrot", params=None, scale=1.0):
        """
        Generate fractal patterns like Mandelbrot, Julia, etc.
        
        Args:
            fractal_type: Type of fractal ('mandelbrot', 'julia', etc.)
            params: Dictionary with specific parameters for the fractal type
            scale: Scaling factor
        """
        self.coordinates = {"x": [], "y": [], "z": []}
        self.colors = []

        # Generate fractal points based on type
        if fractal_type == "mandelbrot":
            self._generate_mandelbrot(params, scale)
        elif fractal_type == "julia":
            self._generate_julia(params, scale)
        elif fractal_type == "burning_ship":
            self._generate_burning_ship(params, scale)
        elif fractal_type == "newton":
            self._generate_newton(params, scale)
        else:
            # Default to simple geometric pattern
            self._generate_default_fractal(params, scale)
    
    def _generate_mandelbrot(self, params, scale):
        """Generate Mandelbrot set points"""
        center = params.get('center', (0.0, 0.0)) if params else (0.0, 0.0)
        zoom = params.get('zoom', 1.0) if params else 1.0
        iterations = params.get('iterations', 100) if params else 100
        
        # Create a grid of complex numbers
        width, height = 100, 100
        x_min, x_max = center[0] - 2/zoom, center[0] + 2/zoom
        y_min, y_max = center[1] - 2/zoom, center[1] + 2/zoom
        
        for i in range(width):
            for j in range(height):
                x = x_min + (x_max - x_min) * i / width
                y = y_min + (y_max - y_min) * j / height
                c = complex(x, y)
                z = 0
                count = 0
                
                while abs(z) < 2 and count < iterations:
                    z = z**2 + c
                    count += 1
                
                # Store point if it's in the set or on the boundary
                if count < iterations or count > iterations * 0.8:
                    self.coordinates["x"].append(x * scale)
                    self.coordinates["y"].append(y * scale)
                    self.coordinates["z"].append(0)
                    color_val = count % 256
                    self.colors.append((color_val, (color_val * 2) % 256, (color_val * 3) % 256))
    
    def _generate_julia(self, params, scale):
        """Generate Julia set points"""
        c_real = params.get('c_real', -0.7) if params else -0.7
        c_imag = params.get('c_imag', 0.27015) if params else 0.27015
        iterations = params.get('iterations', 100) if params else 100
        
        c = complex(c_real, c_imag)
        width, height = 100, 100
        
        for i in range(width):
            for j in range(height):
                x = (i - width/2) * 4.0 / width
                y = (j - height/2) * 4.0 / height
                z = complex(x, y)
                count = 0
                
                while abs(z) < 2 and count < iterations:
                    z = z**2 + c
                    count += 1
                
                if count < iterations or count > iterations * 0.8:
                    self.coordinates["x"].append(x * scale)
                    self.coordinates["y"].append(y * scale)
                    self.coordinates["z"].append(0)
                    color_val = count % 256
                    self.colors.append(((color_val * 3) % 256, color_val, (color_val * 2) % 256))
    
    def _generate_burning_ship(self, params, scale):
        """Generate Burning Ship fractal points"""
        iterations = params.get('iterations', 100) if params else 100
        zoom = params.get('zoom', 1.0) if params else 1.0
        
        width, height = 100, 100
        x_min, x_max = -2.5/zoom, 1.5/zoom
        y_min, y_max = -2.0/zoom, 2.0/zoom
        
        for i in range(width):
            for j in range(height):
                x = x_min + (x_max - x_min) * i / width
                y = y_min + (y_max - y_min) * j / height
                c = complex(x, y)
                z = 0
                count = 0
                
                while abs(z) < 2 and count < iterations:
                    # Burning ship formula: z = (|Re(z)| + i|Im(z)|)^2 + c
                    z = complex(abs(z.real), abs(z.imag))**2 + c
                    count += 1
                
                if count < iterations:
                    self.coordinates["x"].append(x * scale)
                    self.coordinates["y"].append(y * scale)
                    self.coordinates["z"].append(0)
                    color_val = count % 256
                    self.colors.append((color_val, (color_val * 4) % 256, (color_val * 2) % 256))
    
    def _generate_newton(self, params, scale):
        """Generate Newton fractal points"""
        iterations = params.get('iterations', 50) if params else 50
        power = params.get('power', 3) if params else 3
        
        width, height = 100, 100
        
        for i in range(width):
            for j in range(height):
                x = (i - width/2) * 4.0 / width
                y = (j - height/2) * 4.0 / height
                z = complex(x, y)
                count = 0
                
                while abs(z) > 0.001 and count < iterations:
                    # Newton's method for z^n - 1 = 0
                    if abs(z) > 1e-10:
                        z = z - (z**power - 1) / (power * z**(power-1))
                    count += 1
                
                self.coordinates["x"].append(z.real * scale)
                self.coordinates["y"].append(z.imag * scale)
                self.coordinates["z"].append(0)
                color_val = count % 256
                self.colors.append(((color_val * 2) % 256, (color_val * 5) % 256, color_val))
    
    def _generate_default_fractal(self, params, scale):
        """Generate a simple default fractal pattern"""
        # Simple spiral fractal as default
        points = 1000
        for i in range(points):
            t = i * 0.1
            r = scale * np.sqrt(t)
            x = r * np.cos(t)
            y = r * np.sin(t)
            
            self.coordinates["x"].append(x)
            self.coordinates["y"].append(y)
            self.coordinates["z"].append(0)
            color_val = (i * 5) % 256
            self.colors.append((color_val, (color_val * 2) % 256, (color_val * 3) % 256))
    
    def _generate_harmonic_colors(self, values, l, m):
        """Generate colors based on spherical harmonic values"""
        colors = []
        normalized_vals = (np.real(values) - np.min(np.real(values))) / (np.max(np.real(values)) - np.min(np.real(values)) + 1e-8)
        
        for val in normalized_vals:
            # Color based on harmonic degree and order
            hue = (l / 10.0 + abs(m) / 20.0) % 1.0
            saturation = 0.7 + 0.3 * val
            value = 0.5 + 0.5 * val
            colors.append(self._hsv_to_rgb(hue, saturation, value))
        
        return colors
    
    def _generate_hyperbolic_colors(self, x, y, z, function_type):
        """Generate colors based on hyperbolic function values"""
        colors = []
        # Normalize coordinates for coloring
        coord_norm = np.sqrt(x**2 + y**2 + z**2)
        coord_norm = (coord_norm - np.min(coord_norm)) / (np.max(coord_norm) - np.min(coord_norm) + 1e-8)
        
        for i, norm in enumerate(coord_norm):
            if function_type == "sinh":
                hue = 0.1 + 0.3 * norm  # Yellow to orange
            elif function_type == "cosh":
                hue = 0.6 + 0.3 * norm  # Blue to purple
            elif function_type == "tanh":
                hue = 0.3 + 0.3 * norm  # Green to cyan
            else:  # hyperboloid
                hue = 0.8 + 0.2 * norm  # Magenta to red
            
            colors.append(self._hsv_to_rgb(hue % 1.0, 0.8, 0.9))
        
        return colors
    
    def _generate_curve_colors(self, t, curve_type):
        """Generate colors based on curve parameter"""
        colors = []
        t_norm = (t - np.min(t)) / (np.max(t) - np.min(t))
        
        for val in t_norm:
            if curve_type == "lissajous":
                hue = val * 0.7  # Spectrum from red to blue
            elif curve_type == "rose":
                hue = 0.9 - val * 0.3  # Purple to magenta
            elif curve_type == "spiral":
                hue = val * 0.5 + 0.4  # Cyan to blue
            else:  # torus_knot
                hue = val * 0.3 + 0.1  # Orange to yellow
            
            colors.append(self._hsv_to_rgb(hue, 0.9, 0.8))
        
        return colors
    
    def _generate_manifold_colors(self, num_points, manifold_type):
        """Generate colors for manifolds"""
        colors = []
        for i in range(num_points):
            val = i / num_points
            
            if manifold_type == "klein_bottle":
                hue = 0.5 + 0.3 * np.sin(val * 4 * np.pi)
            elif manifold_type == "mobius_strip":
                hue = 0.15 + 0.1 * np.cos(val * 6 * np.pi)
            else:  # boy_surface
                hue = 0.75 + 0.2 * np.sin(val * 8 * np.pi)
            
            colors.append(self._hsv_to_rgb(hue % 1.0, 0.7, 0.8))
        
        return colors
    
    def _hsv_to_rgb(self, h, s, v):
        """Convert HSV to RGB"""
        c = v * s
        x = c * (1 - abs((h * 6) % 2 - 1))
        m = v - c
        
        if h < 1/6:
            r, g, b = c, x, 0
        elif h < 2/6:
            r, g, b = x, c, 0
        elif h < 3/6:
            r, g, b = 0, c, x
        elif h < 4/6:
            r, g, b = 0, x, c
        elif h < 5/6:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))
    
    def project_to_2d(self, projection_type="orthographic", view_angle=(0, 0)):
        """
        Project 3D coordinates to 2D for image rendering
        
        Args:
            projection_type: Type of projection ('orthographic', 'perspective')
            view_angle: Viewing angles (elevation, azimuth) in radians
        """
        if not self.coordinates["x"]:
            return [], []
        
        x, y, z = np.array(self.coordinates["x"]), np.array(self.coordinates["y"]), np.array(self.coordinates["z"])
        
        # Apply rotation
        elevation, azimuth = view_angle
        
        # Rotation matrices
        cos_el, sin_el = np.cos(elevation), np.sin(elevation)
        cos_az, sin_az = np.cos(azimuth), np.sin(azimuth)
        
        # Rotate around y-axis (elevation)
        x_rot = x * cos_el - z * sin_el
        z_rot = x * sin_el + z * cos_el
        
        # Rotate around z-axis (azimuth)
        x_final = x_rot * cos_az - y * sin_az
        y_final = x_rot * sin_az + y * cos_az
        
        if projection_type == "orthographic":
            return x_final, y_final
        elif projection_type == "perspective":
            # Simple perspective projection
            focal_length = 5.0
            x_proj = focal_length * x_final / (focal_length + z_rot + 5)
            y_proj = focal_length * y_final / (focal_length + z_rot + 5)
            return x_proj, y_proj
    
    def render_image(self, image_x, image_y, pad_x, pad_y, projection_type="orthographic", 
                    view_angle=(0.5, 1.0), count=0, render_type="point", white_background=False, return_img=False):
        """
        Render the mathematical function as an image
        
        Args:
            image_x, image_y: Image dimensions
            pad_x, pad_y: Padding
            projection_type: Projection method
            view_angle: Viewing angles
            count: Image count for naming
            render_type: Rendering type ('point' or 'patch')
        """
        if not self.coordinates["x"]:
            print("No coordinates to render")
            return
        
        # Project to 2D
        x_2d, y_2d = self.project_to_2d(projection_type, view_angle)
        
        if len(x_2d) == 0:
            return
        
        # Rescale to image dimensions
        x_min, x_max = np.min(x_2d), np.max(x_2d)
        y_min, y_max = np.min(y_2d), np.max(y_2d)
        
        if x_max == x_min or y_max == y_min:
            return
        
        x_scaled = np.uint16((x_2d - x_min) / (x_max - x_min) * (image_x - 2*pad_x) + pad_x)
        y_scaled = np.uint16((y_2d - y_min) / (y_max - y_min) * (image_y - 2*pad_y) + pad_y)
        
        # Create image with background color
        background_color = 255 if white_background else 0
        image = np.full((image_y, image_x, 3), fill_value=background_color, dtype=np.uint8)
        
        # Render points or patches
        for i in range(len(x_scaled)):
            if 0 <= x_scaled[i] < image_x and 0 <= y_scaled[i] < image_y:
                if render_type == "point":
                    color = self.colors[i] if i < len(self.colors) else (127, 127, 127)
                    image[y_scaled[i], x_scaled[i]] = color
                elif render_type == "patch":
                    # Simple 3x3 patch
                    color = self.colors[i] if i < len(self.colors) else (127, 127, 127)
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            nx, ny = x_scaled[i] + dx, y_scaled[i] + dy
                            if 0 <= nx < image_x and 0 <= ny < image_y:
                                image[ny, nx] = color
        
        # Convert to PIL and optionally return or save
        pil_image = Image.fromarray(image)
        pil_image = pil_image.transpose(Image.FLIP_TOP_BOTTOM)
        
        if return_img:
            # Return image array directly for on-the-fly generation
            img_array = np.array(pil_image)
            pil_image.close()
            return img_array
        else:
            # Original behavior: save with different transformations
            for trans_type in range(4):
                trans_image = self._transpose_image(pil_image, trans_type)
                filename = f"{self.function_name}_{self.function_weight_count}_count_{count}_flip{trans_type}.png"
                trans_image.save(os.path.join(self.save_root, filename))
            
            pil_image.close()
    
    def _transpose_image(self, image, trans_type):
        """Apply image transformations"""
        if trans_type == 0:
            return image.copy()
        elif trans_type == 1:
            return image.transpose(Image.FLIP_TOP_BOTTOM)
        elif trans_type == 2:
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        elif trans_type == 3:
            temp = image.transpose(Image.FLIP_TOP_BOTTOM)
            return temp.transpose(Image.FLIP_LEFT_RIGHT)
        return image.copy()
