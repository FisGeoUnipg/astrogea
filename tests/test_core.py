import os
import pytest
import numpy as np
import xarray as xr
from astrogea.core import process_crism_file
from astrogea.spectral_wrapper import SpectralArrayWrapper
from astrogea.wcs_utils import parse_envi_map_info_list, create_wcs_from_parsed_info

def test_processa_crism_file(tmp_path):
    # Simula input (solo per struttura: usa file veri in un test reale)
    sr_base = "tests/data/frt00006fbd_07_if164j_mtr3.img"
    if_base = "tests/data/frt00006fbd_07_if164j_mtr3.hdr"
    output = tmp_path / "out.nc"

    # Salta se i file non esistono (protezione per CI)
    if not os.path.exists(f"{sr_base}.hdr") or not os.path.exists(f"{if_base}.hdr"):
        import pytest
        pytest.skip("File ENVI non disponibili per il test.")

    ds = process_crism_file(sr_base, if_base, str(output))
    assert isinstance(ds, xr.Dataset)
    assert "surface_reflectance" in ds
    assert "incidence_factor" in ds
    assert output.exists()

def test_spectral_array_wrapper():
    """Test SpectralArrayWrapper class"""
    # Crea dati di test
    data = np.random.rand(10, 5, 5)  # 10 bande, 5x5 pixel
    wavelengths = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
    
    wrapper = SpectralArrayWrapper(data, wavelengths)
    xarray_data = wrapper.to_xarray()
    
    assert isinstance(xarray_data, xr.DataArray)
    assert xarray_data.dims == ('wavelength', 'y', 'x')
    assert len(xarray_data.coords['wavelength']) == 10
    assert xarray_data.shape == (10, 5, 5)

def test_spectral_array_wrapper_invalid_shape():
    """Test SpectralArrayWrapper with invalid data shape"""
    data = np.random.rand(5, 5)  # 2D invece di 3D
    wavelengths = [1.0, 1.1, 1.2, 1.3, 1.4]
    
    with pytest.raises(ValueError, match="Data must be 3D"):
        wrapper = SpectralArrayWrapper(data, wavelengths)
        wrapper.to_xarray()

def test_parse_envi_map_info_list():
    """Test parsing of ENVI map info"""
    map_info = ["UTM", "1.0", "2.0", "100.0", "200.0", "30.0", "30.0", "meters"]
    parsed = parse_envi_map_info_list(map_info)
    
    assert parsed["projection"] == "UTM"
    assert parsed["reference_pixel_x"] == 1.0
    assert parsed["reference_pixel_y"] == 2.0
    assert parsed["map_x"] == 100.0
    assert parsed["map_y"] == 200.0
    assert parsed["pixel_size_x"] == 30.0
    assert parsed["pixel_size_y"] == 30.0
    assert parsed["unit"] == "meters"

def test_create_wcs_from_parsed_info():
    """Test WCS creation from parsed info"""
    parsed_info = {
        "projection": "UTM",
        "reference_pixel_x": 1.0,
        "reference_pixel_y": 2.0,
        "map_x": 100.0,
        "map_y": 200.0,
        "pixel_size_x": 30.0,
        "pixel_size_y": 30.0,
        "unit": "meters"
    }
    shape = (5, 10)  # (rows, cols)
    
    transform = create_wcs_from_parsed_info(parsed_info, shape)
    
    assert transform is not None
    # Verifica che la trasformazione sia corretta
    assert transform.a == 30.0  # pixel_size_x
    assert transform.e == -30.0  # -pixel_size_y (rasterio usa y negativo)

def test_process_crism_file_missing_files(tmp_path):
    """Test process_crism_file with missing input files"""
    with pytest.raises(FileNotFoundError):
        process_crism_file(
            base_sr_path="nonexistent_sr",
            base_if_path="nonexistent_if",
            output_nc_path=str(tmp_path / "output.nc")
        )

def test_process_crism_file_integration(tmp_path):
    """Integration test for process_crism_file (skipped if no real data)"""
    # Simula input (solo per struttura: usa file veri in un test reale)
    sr_base = "tests/data/fake_sr"
    if_base = "tests/data/fake_if"
    output = tmp_path / "out.nc"

    # Salta se i file non esistono (protezione per CI)
    if not os.path.exists(f"{sr_base}.hdr") or not os.path.exists(f"{if_base}.hdr"):
        pytest.skip("File ENVI non disponibili per il test.")

    ds = process_crism_file(sr_base, if_base, str(output))
    assert isinstance(ds, xr.Dataset)
    assert "surface_reflectance" in ds
    assert "incidence_factor" in ds
    assert output.exists()

def test_import_astrogea():
    """Test that astrogea can be imported correctly"""
    import astrogea
    from astrogea import process_crism_file, SpectralArrayWrapper
    
    assert hasattr(astrogea, '__version__')
    assert astrogea.__version__ == "0.1.0"
