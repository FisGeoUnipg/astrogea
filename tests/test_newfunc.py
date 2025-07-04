import numpy as np
import pytest

try:
    import dask.array as da
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

from astrogea.core import (
    remove_crism_bad_ranges_cube, row_norm, column_norm, center_norm, L1_norm, minmax, robust_scaler, derivative, log_1_r_norm,
    baseline_correction_cube, auto_stretch_rgb, unison_shuffled_copies, merge_datacubes, spetial_merge_datacubes, hypermerge_spatial
)

def test_remove_crism_bad_ranges_cube():
    cube = np.random.rand(5, 5, 10)
    wavelengths = np.linspace(1600, 2900, 10)
    cube_good, wavelengths_good = remove_crism_bad_ranges_cube(cube, wavelengths)
    assert cube_good.shape[-1] == len(wavelengths_good)
    assert np.all((wavelengths_good < 1650) | ((wavelengths_good > 1670) & (wavelengths_good < 1930)) | ((wavelengths_good > 2020) & (wavelengths_good < 2650)) | (wavelengths_good > 2820))

def test_row_norm():
    wav = np.linspace(1000, 2000, 10)
    spectra = np.random.rand(5, 10)
    normed = row_norm(wav, spectra)
    assert normed.shape == spectra.shape
    # Each row should integrate to ~1
    for i in range(normed.shape[0]):
        assert np.isclose(np.trapz(normed[i], wav), 1, atol=1e-2)

def test_column_norm():
    spectra = np.random.rand(5, 10)
    normed = column_norm(spectra)
    assert normed.shape == spectra.shape
    # Each column should have mean ~0 and std ~1
    for i in range(normed.shape[1]):
        assert np.isclose(np.mean(normed[:, i]), 0, atol=1e-2)
        assert np.isclose(np.std(normed[:, i]), 1, atol=1e-2)

def test_center_norm():
    spectra = np.random.rand(5, 10)
    normed = center_norm(spectra)
    for i in range(normed.shape[1]):
        assert np.isclose(np.mean(normed[:, i]), 0, atol=1e-2)

def test_L1_norm():
    spectra = np.random.rand(5, 10)
    normed = L1_norm(spectra)
    for i in range(normed.shape[1]):
        assert np.isclose(np.linalg.norm(normed[:, i], ord=1), 1, atol=1e-2)

def test_minmax():
    spectra = np.random.rand(5, 10)
    normed = minmax(spectra, mode='zero to one')
    assert np.all(normed >= 0) and np.all(normed <= 1)
    normed2 = minmax(spectra, mode='-one to one')
    assert np.all(normed2 >= -1) and np.all(normed2 <= 1)

def test_baseline_correction_cube():
    cube = np.random.rand(3, 3, 8)
    wavelengths = np.linspace(1000, 2000, 8)
    corrected = baseline_correction_cube(cube, wavelengths, order=1)
    assert corrected.shape == cube.shape

def test_log_1_r_norm():
    spectra = np.random.rand(5, 10) + 0.1  # avoid zeros
    normed = log_1_r_norm(spectra)
    assert np.all(np.isfinite(normed))

def test_unison_shuffled_copies():
    spectra = np.arange(10).reshape(5, 2)
    indexes = np.arange(5)
    shuffled_spectra, shuffled_indexes = unison_shuffled_copies(spectra, indexes)
    assert set(shuffled_indexes) == set(indexes)
    assert set(tuple(row) for row in shuffled_spectra) == set(tuple(row) for row in spectra)

def test_merge_datacubes():
    cube1 = np.random.rand(2, 2, 3)
    cube2 = np.random.rand(2, 2, 3)
    merged = merge_datacubes([cube1, cube2], axis=0)
    assert merged.shape[0] == cube1.shape[0] + cube2.shape[0]

def test_hypermerge_spatial():
    cube1 = np.random.rand(2, 2, 3)
    cube2 = np.random.rand(2, 2, 3)
    merged = hypermerge_spatial([cube1, cube2])
    assert merged.shape == cube1.shape

@pytest.mark.skipif(not DASK_AVAILABLE, reason="Dask not available")
def test_L1_norm_dask():
    spectra = np.random.rand(5, 10)
    spectra_dask = da.from_array(spectra, chunks=(5, 10))
    normed = L1_norm(spectra_dask, use_dask=True)
    normed_np = normed.compute()
    for i in range(normed_np.shape[1]):
        assert np.isclose(np.linalg.norm(normed_np[:, i], ord=1), 1, atol=1e-2) 