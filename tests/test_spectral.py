import numpy as np
from astrogea.core import coregister_spectra

try:
    import dask.array as da
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

def test_coregister_spectra_linear():
    reference_wavelengths = np.array([1, 2, 3, 4, 5])
    new_wavelengths = np.array([1, 3, 5])
    new_reflectance = np.array([10, 30, 50])
    # L'interpolazione lineare tra i punti deve restituire [10, 20, 30, 40, 50]
    expected = np.array([10, 20, 30, 40, 50])
    # Test con NumPy
    result = coregister_spectra(reference_wavelengths, new_wavelengths, new_reflectance)
    np.testing.assert_allclose(result, expected)
    # Test con Dask (se disponibile)
    if DASK_AVAILABLE:
        d_reference_wavelengths = da.from_array(reference_wavelengths, chunks=5)
        d_new_wavelengths = da.from_array(new_wavelengths, chunks=3)
        d_new_reflectance = da.from_array(new_reflectance, chunks=3)
        # coregister_spectra deve funzionare anche con input Dask (convertiti a NumPy internamente)
        d_result = coregister_spectra(d_reference_wavelengths, d_new_wavelengths, d_new_reflectance)
        # Se il risultato è Dask, computa
        if hasattr(d_result, 'compute'):
            d_result = d_result.compute()
        np.testing.assert_allclose(d_result, expected) 