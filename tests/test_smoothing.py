import numpy as np
import pytest

from astrogea.core import smoothing_moving_average

try:
    import dask.array as da
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

def test_smoothing_numpy():
    img = np.ones((2, 2, 5))
    wavelength = np.arange(5)
    MIN, MAX = 1, 4
    window_size = 3
    # L'output atteso è sempre 1, perché la media di tutti 1 è 1
    result, x = smoothing_moving_average(img, wavelength, MIN, MAX, window_size=window_size)
    assert result.shape == (2, 2, 3)
    np.testing.assert_allclose(result, 1)
    np.testing.assert_allclose(x, [1,2,3])

@pytest.mark.skipif(not DASK_AVAILABLE, reason="Dask non disponibile")
def test_smoothing_dask():
    img = np.ones((2, 2, 5))
    wavelength = np.arange(5)
    MIN, MAX = 1, 4
    window_size = 3
    dimg = da.from_array(img, chunks=(1, 1, 5))
    result, x = smoothing_moving_average(dimg, wavelength, MIN, MAX, window_size=window_size, use_dask=True)
    result_np = result.compute()
    assert result_np.shape == (2, 2, 3)
    np.testing.assert_allclose(result_np, 1)
    np.testing.assert_allclose(x, [1,2,3]) 