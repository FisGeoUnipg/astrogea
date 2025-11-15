#!/bin/bash

# Minimal setup for astrogea
set -e

echo "=== Astrogea Setup - Minimal Version ==="

# 1. Create virtual environment
echo "1. Creating virtual environment..."
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install base dependencies
echo "2. Installing dependencies..."
pip install --upgrade pip
pip install numpy xarray spectral astropy scipy matplotlib
pip install dask[complete] netCDF4

# 3. Install astrogea in development mode
echo "3. Installing astrogea..."
pip install -e .

# 4. Create required directories
echo "4. Creating directories..."
mkdir -p data output logs

# 5. Verify installation
echo "5. Verifying installation..."
python -c "import astrogea; print('astrogea installed correctly')"

echo ""
echo "=== Setup Completed! ==="
echo ""
echo "To run the example:"
echo "  source venv/bin/activate"
echo "  python examples/simple_example.py"
echo ""
echo "For Windows:"
echo "  venv\\Scripts\\activate"
echo "  python examples\\simple_example.py"

