#!/bin/bash

# Enhanced FractalDB Development Environment Setup Script
set -e

echo "🚀 Setting up Enhanced FractalDB Development Environment..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install system dependencies for scientific computing and graphics
echo "🔧 Installing system dependencies..."
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    htop \
    tree \
    jq \
    unzip \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release

# Install graphics and image processing libraries
echo "🖼️  Installing graphics and image processing dependencies..."
sudo apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libfontconfig1 \
    libglu1-mesa \
    freeglut3-dev \
    libglew-dev \
    libglfw3-dev \
    libglm-dev \
    libao-dev \
    libmpg123-dev \
    xvfb

# Install Python development tools
echo "🐍 Installing Python development tools..."
sudo apt-get install -y \
    python3-dev \
    python3-pip \
    python3-venv \
    python3-wheel \
    python3-setuptools

# Install additional dependencies for mathematical operations
echo "🔢 Installing mathematical libraries..."
sudo apt-get install -y \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    pkg-config \
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev

# Install dependencies for PDF generation
echo "📄 Installing PDF generation tools..."
sudo apt-get install -y \
    pandoc \
    texlive-latex-base \
    texlive-latex-recommended \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    lmodern

# Create virtual display for matplotlib rendering
echo "🖥️  Setting up virtual display..."
sudo apt-get install -y xvfb
export DISPLAY=:99
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &

# Set up Python environment
echo "🐍 Setting up Python environment..."
pip3 install --upgrade pip setuptools wheel

# Install common data science and development tools
echo "📊 Installing development tools..."
pip3 install \
    jupyter \
    jupyterlab \
    ipykernel \
    black \
    isort \
    pylint \
    flake8 \
    pytest \
    pre-commit

# Install additional useful packages for development
echo "🛠️  Installing additional development packages..."
pip3 install \
    tqdm \
    rich \
    click \
    python-dotenv \
    requests \
    httpx

# Configure git (if not already configured)
echo "📝 Configuring git..."
if [ -z "$(git config --global user.name)" ]; then
    git config --global user.name "Codespace User"
fi
if [ -z "$(git config --global user.email)" ]; then
    git config --global user.email "user@codespace.local"
fi

# Set up pre-commit hooks
echo "🪝 Setting up pre-commit hooks..."
if [ -f ".pre-commit-config.yaml" ]; then
    pre-commit install
fi

# Create helpful aliases
echo "🔗 Creating helpful aliases..."
cat >> ~/.bashrc << 'EOF'

# Enhanced FractalDB Development Aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias ..='cd ..'
alias ...='cd ../..'
alias grep='grep --color=auto'
alias fgrep='fgrep --color=auto'
alias egrep='egrep --color=auto'

# Project specific aliases
alias demo='python demo_enhanced_functions.py'
alias jupyter-lab='jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root'
alias serve='python -m http.server 8000'

# Git aliases
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git log --oneline'
alias gd='git diff'

EOF

# Create a welcome message
echo "📄 Creating welcome message..."
cat > ~/.motd << 'EOF'
╭─────────────────────────────────────────────────────────────╮
│                                                             │
│   🎨 Enhanced FractalDB Development Environment 🎨          │
│                                                             │
│   Welcome to your fractal rendering and ML workspace!      │
│                                                             │
│   Quick Start Commands:                                     │
│   • demo                    - Run the demo script          │
│   • jupyter-lab            - Start Jupyter Lab             │
│   • serve                   - Start HTTP server            │
│                                                             │
│   Useful directories:                                       │
│   • demo_output/           - Generated images              │
│   • fractal_renderer/      - Core rendering library        │
│                                                             │
╰─────────────────────────────────────────────────────────────╯
EOF

# Add welcome message to bashrc
echo 'cat ~/.motd' >> ~/.bashrc

# Clean up
echo "🧹 Cleaning up..."
sudo apt-get autoremove -y
sudo apt-get autoclean

echo "✅ Enhanced FractalDB Development Environment setup complete!"
echo "🎯 Ready to create amazing fractals and train neural networks!"

# Display system info
echo ""
echo "📋 System Information:"
echo "Python version: $(python3 --version)"
echo "Pip version: $(pip3 --version)"
echo "Available CPU cores: $(nproc)"
echo "Available memory: $(free -h | awk '/^Mem:/ {print $2}')"
echo ""
echo "🚀 Environment is ready! Start coding with 'python demo_enhanced_functions.py'"
