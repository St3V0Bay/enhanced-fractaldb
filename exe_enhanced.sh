#!/bin/bash

# Enhanced FractalDB Execution Script
# Demonstrates both original fractal generation and new mathematical functions

echo "Enhanced FractalDB - Mathematical Functions & Fractals"
echo "===================================================="

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)/fractal_renderer"

# Check for required dependencies
python3 -c "import numpy, scipy, PIL" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required dependencies..."
    pip3 install numpy scipy pillow
fi

echo ""
echo "Step 1: Running demonstration of mathematical functions..."
echo "--------------------------------------------------------"
python3 demo_enhanced_functions.py

echo ""
echo "Step 2: Generate small-scale enhanced FractalDB (mathematical functions only)..."
echo "-----------------------------------------------------------------------------"

# Create necessary directories
mkdir -p data
mkdir -p fractal_renderer/weights

# Create a simple weight file if it doesn't exist
if [ ! -f "fractal_renderer/weights/weights_0.1.csv" ]; then
    echo "Creating default weights file..."
    cat > fractal_renderer/weights/weights_0.1.csv << 'EOF'
1.0,0.0,0.0,1.0,0.0,0.0
0.5,0.5,-0.5,0.5,0.25,0.25
0.5,-0.5,0.5,0.5,0.5,0.0
EOF
fi

# Run enhanced FractalDB generation with mathematical functions
python3 fractal_renderer/make_enhanced_fractaldb.py \
    --save_root="./data/EnhancedFractalDB-Demo" \
    --enable_math_functions=True \
    --math_categories=20 \
    --instance=2 \
    --image_size_x=256 \
    --image_size_y=256 \
    --draw_type="point_color" \
    --spherical_harmonics_ratio=0.25 \
    --hyperbolic_functions_ratio=0.25 \
    --parametric_curves_ratio=0.25 \
    --complex_manifolds_ratio=0.25

echo ""
echo "Step 3: Generate original fractal categories (if CSV data available)..."
echo "---------------------------------------------------------------------"

# Check if we have fractal category data
if [ -d "data" ] && [ "$(ls -A data/*.csv 2>/dev/null)" ]; then
    echo "Found fractal CSV files, generating fractal images..."
    python3 fractal_renderer/make_fractaldb.py \
        --load_root="./data" \
        --save_root="./data/OriginalFractals-Demo" \
        --instance=2 \
        --image_size_x=256 \
        --image_size_y=256 \
        --draw_type="point_color"
else
    echo "No fractal CSV files found. To generate original fractals, first run:"
    echo "  python param_search/ifs_search.py --rate=0.2 --category=10 --numof_point=10000 --save_dir='./data'"
fi

echo ""
echo "Step 4: Analyzing generated datasets..."
echo "--------------------------------------"

# Count generated images
if [ -d "data/EnhancedFractalDB-Demo" ]; then
    math_categories=$(find data/EnhancedFractalDB-Demo -type d -name "*_*" | wc -l)
    math_images=$(find data/EnhancedFractalDB-Demo -name "*.png" | wc -l)
    echo "Enhanced Mathematical Functions:"
    echo "  Categories: $math_categories"
    echo "  Total images: $math_images"
fi

if [ -d "data/OriginalFractals-Demo" ]; then
    fractal_categories=$(find data/OriginalFractals-Demo -type d -name "*" -not -path "data/OriginalFractals-Demo" | wc -l)
    fractal_images=$(find data/OriginalFractals-Demo -name "*.png" | wc -l)
    echo "Original Fractals:"
    echo "  Categories: $fractal_categories"
    echo "  Total images: $fractal_images"
fi

if [ -d "demo_output" ]; then
    demo_images=$(find demo_output -name "*.png" | wc -l)
    echo "Demo images: $demo_images"
fi

echo ""
echo "Step 5: Summary of Mathematical Functions"
echo "----------------------------------------"
echo "The enhanced FractalDB now includes:"
echo ""
echo "1. SPHERICAL HARMONICS (Y_l^m)"
echo "   • Mathematical foundation: Solutions to Laplace equation on sphere"
echo "   • Used in: Quantum mechanics, computer graphics, climate modeling"
echo "   • Parameters: degree l, order m, resolution, radius modulation"
echo ""
echo "2. HYPERBOLIC FUNCTIONS"
echo "   • Types: sinh, cosh, tanh surfaces, hyperboloids"
echo "   • Mathematical foundation: Non-Euclidean geometry"
echo "   • Used in: Relativity, differential geometry, neural network activations"
echo ""
echo "3. N-DIMENSIONAL PARAMETRIC CURVES"
echo "   • Types: Lissajous, rose curves, spirals, torus knots"
echo "   • Mathematical foundation: Parametric equations in 3D space"
echo "   • Used in: Oscillation analysis, robotics, knot theory"
echo ""
echo "4. COMPLEX MANIFOLDS"
echo "   • Types: Klein bottle, Möbius strip, Boy's surface"
echo "   • Mathematical foundation: Differential topology"
echo "   • Used in: Abstract algebra, theoretical physics"
echo ""
echo "Benefits for Neural Network Training:"
echo "• Increased geometric diversity beyond traditional fractals"
echo "• Rich mathematical structures with inherent symmetries"
echo "• Scalable complexity through parameter variation"
echo "• Natural data augmentation through viewing angles and projections"
echo "• Theoretical foundation for understanding learned representations"

echo ""
echo "Execution completed! Check the following directories:"
echo "• demo_output/ - Sample mathematical function visualizations"
echo "• data/EnhancedFractalDB-Demo/ - Generated mathematical function dataset"
if [ -d "data/OriginalFractals-Demo" ]; then
    echo "• data/OriginalFractals-Demo/ - Generated fractal dataset"
fi
echo ""
echo "To generate a full-scale dataset for training, use:"
echo "python3 fractal_renderer/make_enhanced_fractaldb.py --math_categories=1000 --instance=10"
