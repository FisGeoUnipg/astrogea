import numpy as np
from astrogea.core import continuum_removal, band_parameters_mafic

try:
    import dask.array as da
    has_dask = True
except ImportError:
    has_dask = False

# Dati sintetici: cubo 4x4x10, valori random positivi
np.random.seed(42)
img = np.random.rand(4, 4, 10) * 1000
wavelength = np.linspace(1000, 2000, 10)
MIN, MAX = 1100, 1900

# --- NumPy ---
print("--- Analisi con NumPy ---")
result, x = continuum_removal(img, wavelength, MIN, MAX)
mafic_map = band_parameters_mafic(result, x, nbands=3)
print("Mappa parametri mafic (NumPy):\n", mafic_map)

# --- Dask ---
if has_dask:
    print("--- Analisi con Dask ---")
    dask_img = da.from_array(img, chunks=(2, 2, 10))
    result_dask, x_dask = continuum_removal(dask_img, wavelength, MIN, MAX, use_dask=True)
    result_dask = result_dask.compute()
    mafic_map_dask = band_parameters_mafic(result_dask, x_dask, nbands=3, use_dask=False)
    print("Mappa parametri mafic (Dask, calcolo NumPy):\n", mafic_map_dask)
    # Analisi completamente Dask
    mafic_map_dask2 = band_parameters_mafic(dask_img, wavelength, nbands=3, use_dask=True)
    print("Mappa parametri mafic (Dask, calcolo Dask):\n", mafic_map_dask2.compute())
else:
    print("Dask non disponibile, test solo NumPy.") 