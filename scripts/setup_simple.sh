#!/bin/bash

# Setup minimale per astrogea
set -e

echo "=== Setup Astrogea - Versione Minimale ==="

# 1. Crea ambiente virtuale
echo "1. Creazione ambiente virtuale..."
python -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate

# 2. Installa dipendenze base
echo "2. Installazione dipendenze..."
pip install --upgrade pip
pip install numpy xarray spectral astropy scipy matplotlib
pip install dask[complete] netCDF4

# 3. Install astrogea in development mode
echo "3. Installazione astrogea..."
pip install -e .

# 4. Crea directory necessarie
echo "4. Creazione directory..."
mkdir -p data output logs

# 5. Verifica installazione
echo "5. Verifica installazione..."
python -c "import astrogea; print('✅ astrogea installato correttamente')"

echo ""
echo "=== Setup Completato! ==="
echo ""
echo "Per eseguire l'esempio:"
echo "  source venv/bin/activate"
echo "  python examples/simple_example.py"
echo ""
echo "Per Windows:"
echo "  venv\\Scripts\\activate"
echo "  python examples\\simple_example.py"

