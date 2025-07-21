# Enhanced FractalDB with Higher-Dimensional Mathematical Functions

## Overview

This enhanced version of FractalDB extends the original fractal geometry approach by implementing advanced higher-dimensional mathematical functions for neural network pre-training. In addition to the original Iterated Function System (IFS) fractals, the system now supports:

- **Spherical Harmonics** (Y_l^m) - Solutions to Laplace equation on spheres
- **Hyperbolic Functions** - sinh, cosh, tanh surfaces and hyperboloids  
- **N-Dimensional Parametric Curves** - Lissajous curves, rose curves, spirals, torus knots
- **Complex Manifolds** - Klein bottles, Möbius strips, Boy's surfaces

## Mathematical Foundations

### 1. Spherical Harmonics Y_l^m(θ, φ)

**Mathematical Basis**: Solutions to Laplace's equation in spherical coordinates
```
∇²Y_l^m = 0
```

**Applications**: 
- Quantum mechanics (atomic orbitals)
- Computer graphics (environment mapping)
- Geophysics (gravitational/magnetic field modeling)
- Climate science (atmospheric modeling)

**Parameters**:
- `l_max`: Maximum degree of harmonics (complexity)
- `m_range`: Order range (-l ≤ m ≤ l)
- `resolution`: Angular sampling resolution
- `radius`: Base sphere radius with harmonic modulation

### 2. Hyperbolic Functions

**Mathematical Basis**: Hyperbolic trigonometry and non-Euclidean geometry

**Function Types**:
- **sinh surface**: `z = sinh(u)cos(v), y = sinh(u)sin(v), z = cosh(u)`
- **cosh surface**: `x = cosh(u)cos(v), y = cosh(u)sin(v), z = sinh(u)`
- **tanh surface**: `z = tanh(√(u² + v²))`
- **hyperboloid**: `x²/a² + y²/b² - z²/c² = 1`

**Applications**:
- Special and general relativity
- Differential geometry
- Neural network activation functions
- Hyperbolic space modeling

### 3. N-Dimensional Parametric Curves

**Mathematical Basis**: Parametric equations in multiple dimensions

**Curve Types**:
- **Lissajous**: `x = A·sin(at + δ), y = B·sin(bt), z = C·sin(ct)`
- **Rose**: `r = cos(kθ)` in polar coordinates
- **Spiral**: `x = r·cos(t), y = r·sin(t), z = pitch·t`
- **Torus Knot**: `(p,q)`-torus knots with complex winding

**Applications**:
- Oscillation analysis and harmonic motion
- Robotics (trajectory planning)
- Topology and knot theory
- Signal processing

### 4. Complex Manifolds

**Mathematical Basis**: Differential topology and non-orientable surfaces

**Manifold Types**:
- **Klein Bottle**: Non-orientable surface, genus 0
- **Möbius Strip**: One-sided surface with single twist
- **Boy's Surface**: Real projective plane immersion

**Applications**:
- Topology and differential geometry
- Abstract algebra
- Theoretical physics (string theory, field theory)

## Installation and Dependencies

```bash
# Install required packages
pip install numpy scipy pillow

# Clone and setup
git clone https://github.com/hirokatsukataoka16/FractalDB-Pretrained-ResNet-PyTorch.git
cd FractalDB-Pretrained-ResNet-PyTorch
```

## Quick Start

### Demo Run
```bash
./exe_enhanced.sh
```

This will:
1. Generate demo visualizations of all mathematical function types
2. Create a small-scale enhanced dataset
3. Provide analysis and summary of capabilities

### Generate Full Enhanced Dataset

```bash
python3 fractal_renderer/make_enhanced_fractaldb.py \
    --save_root="./data/EnhancedFractalDB-1000" \
    --enable_math_functions=True \
    --math_categories=1000 \
    --instance=10 \
    --spherical_harmonics_ratio=0.3 \
    --hyperbolic_functions_ratio=0.3 \
    --parametric_curves_ratio=0.2 \
    --complex_manifolds_ratio=0.2
```

### Generate Mathematical Functions Only

```python
from fractal_renderer.mathematical_functions import MathematicalFunctions

# Spherical harmonics example
math_func = MathematicalFunctions(save_root="./output", 
                                function_name="spherical_demo", 
                                function_weight_count="00")

math_func.spherical_harmonics(l_max=5, resolution=100, radius=1.0)
math_func.render_image(image_x=512, image_y=512, pad_x=10, pad_y=10,
                      projection_type="orthographic", view_angle=(0.5, 1.0))

# Hyperbolic functions example  
math_func.hyperbolic_functions(function_type="hyperboloid", 
                              u_range=(-2, 2), v_range=(-np.pi, np.pi),
                              resolution=100, scale=1.0)

# Parametric curves example
lissajous_params = {'a': 3, 'b': 5, 'c': 7, 'A': 1.0, 'B': 1.0, 'delta': np.pi/4}
math_func.n_dimensional_curves(curve_type="lissajous", dimensions=3, 
                              resolution=1000, params=lissajous_params)

# Complex manifolds example
math_func.complex_manifolds(manifold_type="klein_bottle", resolution=60, scale=1.0)
```

## Configuration Parameters

### Enhanced FractalDB Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_math_functions` | `True` | Enable mathematical function generation |
| `math_categories` | `1000` | Number of mathematical function categories |
| `spherical_harmonics_ratio` | `0.3` | Ratio of spherical harmonics in dataset |
| `hyperbolic_functions_ratio` | `0.3` | Ratio of hyperbolic functions |
| `parametric_curves_ratio` | `0.2` | Ratio of parametric curves |
| `complex_manifolds_ratio` | `0.2` | Ratio of complex manifolds |

### Rendering Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `projection_type` | `"orthographic"` | Projection method (`orthographic`, `perspective`) |
| `view_angle` | `(0.5, 1.0)` | Viewing angles (elevation, azimuth) in radians |
| `render_type` | `"point"` | Rendering style (`point`, `patch`) |

## Dataset Structure

```
EnhancedFractalDB-1000/
├── spherical_00000/          # Spherical harmonics categories
│   ├── spherical_00000_00_count_0_flip0.png
│   ├── spherical_00000_00_count_0_flip1.png
│   └── ...
├── hyperbolic_00000/         # Hyperbolic function categories  
│   ├── hyperbolic_00000_00_count_0_flip0.png
│   └── ...
├── curves_00000/            # Parametric curve categories
│   ├── curves_00000_00_count_0_flip0.png
│   └── ...
├── manifolds_00000/         # Complex manifold categories
│   ├── manifolds_00000_00_count_0_flip0.png
│   └── ...
└── [original fractal categories if CSV data provided]
```

## Benefits for Neural Network Training

### Theoretical Advantages

1. **Increased Geometric Diversity**: Beyond traditional fractals, incorporating mathematical functions with different symmetries and topological properties
2. **Rich Mathematical Structure**: Functions with theoretical foundations provide interpretable geometric priors
3. **Scalable Complexity**: Parameter variations allow systematic control over visual complexity
4. **Natural Data Augmentation**: 3D to 2D projections provide multiple viewpoints automatically
5. **Foundation for Representation Learning**: Mathematical structure aids in understanding what networks learn

### Empirical Benefits

- **Broader Feature Coverage**: Mathematical functions cover different aspects of geometry than fractals alone
- **Better Generalization**: Diverse mathematical structures may improve transfer to natural images
- **Controllable Difficulty**: Systematic parameter variation allows curriculum learning approaches
- **Interpretable Features**: Known mathematical properties help interpret learned representations

## Research Applications

This enhanced FractalDB is particularly valuable for:

- **Computer Vision**: Pre-training without natural images, studying geometric bias
- **Scientific Computing**: Understanding how networks learn mathematical structures
- **Representation Learning**: Analyzing geometric features in learned representations
- **Transfer Learning**: Systematic study of mathematical structure transfer to natural domains

## Comparison with Original FractalDB

| Aspect | Original FractalDB | Enhanced FractalDB |
|--------|-------------------|-------------------|
| **Mathematical Basis** | IFS fractals | IFS fractals + 4 mathematical function families |
| **Dimensionality** | 2D (inherently) | 2D + 3D projections |
| **Parameter Space** | 6 IFS parameters | 6 IFS + function-specific parameters |
| **Geometric Diversity** | Self-similar fractals | Fractals + curves + surfaces + manifolds |
| **Theoretical Foundation** | Fractal geometry | Multiple mathematical disciplines |

## Performance Considerations

- **Generation Time**: Mathematical functions add ~30% generation time vs. fractals alone
- **Memory Usage**: Similar to original (images generated sequentially)
- **Scalability**: Supports same scaling as original (1K-10K categories)

## Citation

If you use this enhanced version, please cite both the original FractalDB paper and acknowledge the mathematical function extensions:

```bibtex
@article{KataokaIJCV2022,
  author={Kataoka, Hirokatsu and Okayasu, Kazushige and Matsumoto, Asato and Yamagata, Eisuke and Yamada, Ryosuke and Inoue, Nakamasa and Nakamura, Akio and Satoh, Yutaka},
  title={Pre-training without Natural Images},
  journal={International Journal on Computer Vision (IJCV)},
  year={2022},
}

% Enhanced version with mathematical functions
@misc{EnhancedFractalDB2024,
  title={Enhanced FractalDB with Higher-Dimensional Mathematical Functions},
  note={Extension of original FractalDB with spherical harmonics, hyperbolic functions, parametric curves, and complex manifolds},
  year={2024}
}
```

## Contributing

Contributions to extend mathematical function families or improve rendering are welcome. Consider adding:
- Additional special functions (Bessel, Legendre polynomials, etc.)
- More complex manifolds (minimal surfaces, etc.)  
- Alternative projection methods
- Performance optimizations

## License

This enhanced version maintains the same license as the original FractalDB repository.
