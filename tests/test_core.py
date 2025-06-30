import os
import pytest
import numpy as np
import xarray as xr
from astrogea.core import process_crism_file, continuum_removal, continuum_to_xarray_wcs
from astrogea.spectral_wrapper import SpectralArrayWrapper
from astrogea.wcs_utils import parse_envi_map_info_list, create_wcs_from_parsed_info
import pandas as pd

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
    map_info = [
        "UTM",  # projection_name
        1.0,    # ref_x_1based
        2.0,    # ref_y_1based
        100.0,  # ref_coord1
        200.0,  # ref_coord2
        30.0,   # pixel_size_x
        30.0,   # pixel_size_y
        "meters"  # units
    ]
    parsed = parse_envi_map_info_list(map_info)
    assert parsed["projection_name"] == "UTM"
    assert parsed["ref_x_1based"] == 1.0
    assert parsed["ref_y_1based"] == 2.0
    assert parsed["ref_coord1"] == 100.0
    assert parsed["ref_coord2"] == 200.0
    assert parsed["pixel_size_x"] == 30.0
    assert parsed["pixel_size_y"] == 30.0
    assert parsed["units"] in ("meters", "m")
    assert "datum" in parsed

def test_create_wcs_from_parsed_info():
    """Test WCS creation from parsed info"""
    parsed_info = {
        "projection": "UTM",
        "ref_x_1based": 1.0,
        "ref_y_1based": 2.0,
        "map_x": 100.0,
        "map_y": 200.0,
        "pixel_size_x": 30.0,
        "pixel_size_y": 30.0,
        "unit": "meters"  # unità supportata
    }
    shape = (5, 10)  # (rows, cols)
    transform = create_wcs_from_parsed_info(parsed_info, shape)
    if transform is None:
        import pytest
        pytest.skip("WCS non creato: controlla che astropy sia installato e che i dati siano corretti.")
    assert hasattr(transform, 'a') and transform.a == 30.0  # pixel_size_x
    assert hasattr(transform, 'e') and transform.e == -30.0  # -pixel_size_y (rasterio usa y negativo)

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
    assert astrogea.__version__ == "0.1.1"

def test_continuum_removal_numpy():
    from astrogea.core import continuum_removal
    # Dati sintetici: cubo 4x4x10, valori random positivi
    np.random.seed(42)
    img = np.random.rand(4, 4, 10) * 1000
    wavelength = np.linspace(1000, 2000, 10)
    MIN, MAX = 1200, 1800
    result, x = continuum_removal(img, wavelength, MIN, MAX)
    assert result.shape == (4, 4, len(x))
    assert np.all(np.isfinite(result))
    # Il continuo normalizzato deve essere vicino a 1 in assenza di assorbimenti
    assert np.nanmax(result) <= 1.5
    assert np.nanmin(result) >= 0

def test_continuum_removal_dask():
    try:
        import dask.array as da
    except ImportError:
        import pytest
        pytest.skip("Dask non disponibile")
    from astrogea.core import continuum_removal
    np.random.seed(42)
    img = np.random.rand(4, 4, 10) * 1000
    dask_img = da.from_array(img, chunks=(2, 2, 10))
    wavelength = np.linspace(1000, 2000, 10)
    MIN, MAX = 1200, 1800
    result, x = continuum_removal(dask_img, wavelength, MIN, MAX, use_dask=True)
    # Il risultato deve essere un dask array
    assert hasattr(result, 'compute')
    computed = result.compute()
    assert computed.shape == (4, 4, len(x))
    assert np.all(np.isfinite(computed))
    assert np.nanmax(computed) <= 1.5
    assert np.nanmin(computed) >= 0

def test_continuum_removal_to_xarray_and_save(tmp_path):
    # Dati sintetici
    img = np.random.rand(3, 4, 8) * 1000
    wavelength = np.linspace(1000, 2000, 8)
    MIN, MAX = 1100, 1900
    result, x = continuum_removal(img, wavelength, MIN, MAX)
    # Mappa info fittizia
    map_info = [
        "UTM", 1.0, 2.0, 100.0, 200.0, 30.0, 30.0, "meters"
    ]
    parsed_map_info = parse_envi_map_info_list(map_info)
    da = continuum_to_xarray_wcs(result, x, parsed_map_info)
    # Salva in NetCDF nella directory temporanea
    out_path_tmp = tmp_path / "out.nc"
    da.to_netcdf(out_path_tmp)
    assert out_path_tmp.exists()
    # Salva anche nella directory corrente per verifica manuale
    out_path_local = os.path.abspath("out.nc")
    da.to_netcdf(out_path_local)
    print(f"File risultato salvato in: {out_path_local}")
    # Riapri e verifica attributi
    loaded = xr.open_dataarray(out_path_tmp)
    assert "wcs_header_dict" in loaded.attrs or loaded.attrs.get("has_wcs", 0) == 1

def test_continuum_removal_save_netcdf4_with_wcs():
    import numpy as np
    import xarray as xr
    import os
    from astrogea.core import continuum_removal, continuum_to_xarray_wcs
    from astrogea.wcs_utils import parse_envi_map_info_list
    # Dati sintetici
    img = np.random.rand(3, 4, 8) * 1000
    wavelength = np.linspace(1000, 2000, 8)
    MIN, MAX = 1100, 1900
    map_info = [
        "UTM", 1.0, 2.0, 100.0, 200.0, 30.0, 30.0, "meters"
    ]
    parsed_map_info = parse_envi_map_info_list(map_info)
    # --- NumPy ---
    result, x = continuum_removal(img, wavelength, MIN, MAX)
    da = continuum_to_xarray_wcs(result, x, parsed_map_info)
    out_path = os.path.abspath("out_continuum_numpy.nc")
    da.to_netcdf(out_path, engine="netcdf4")
    print(f"File NetCDF4 (NumPy) salvato: {out_path}")
    loaded = xr.open_dataarray(out_path)
    assert "wcs_header_dict" in loaded.attrs or loaded.attrs.get("has_wcs", 0) == 1
    # --- Dask ---
    try:
        import dask.array as da_dask
        dask_img = da_dask.from_array(img, chunks=(2, 2, 8))
        result_dask, x_dask = continuum_removal(dask_img, wavelength, MIN, MAX, use_dask=True)
        computed = result_dask.compute()
        da_dask_xr = continuum_to_xarray_wcs(computed, x_dask, parsed_map_info)
        out_path_dask = os.path.abspath("out_continuum_dask.nc")
        da_dask_xr.to_netcdf(out_path_dask, engine="netcdf4")
        print(f"File NetCDF4 (Dask) salvato: {out_path_dask}")
        loaded_dask = xr.open_dataarray(out_path_dask)
        assert "wcs_header_dict" in loaded_dask.attrs or loaded_dask.attrs.get("has_wcs", 0) == 1
    except ImportError:
        print("Dask non disponibile, file Dask non creato.")

def test_continuum_removal_save_netcdf3_textual_with_wcs():
    import numpy as np
    import xarray as xr
    import os
    import string
    from astrogea.core import continuum_removal, continuum_to_xarray_wcs
    from astrogea.wcs_utils import parse_envi_map_info_list
    # Dati sintetici
    img = np.random.rand(3, 4, 8) * 1000
    wavelength = np.linspace(1000, 2000, 8)
    MIN, MAX = 1100, 1900
    map_info = [
        "UTM", 1.0, 2.0, 100.0, 200.0, 30.0, 30.0, "meters"
    ]
    parsed_map_info = parse_envi_map_info_list(map_info)
    result, x = continuum_removal(img, wavelength, MIN, MAX)
    da = continuum_to_xarray_wcs(result, x, parsed_map_info)
    out_path = os.path.abspath("out_continuum_textual.nc")
    da.to_netcdf(out_path, engine="scipy", format="NETCDF3_CLASSIC")
    print(f"File NetCDF3 (testuale) salvato: {out_path}")
    # Verifica che il file sia stato creato
    assert os.path.exists(out_path)
    # Stampa solo l'header ASCII del file
    with open(out_path, "rb") as f:
        header = f.read(1024)
        ascii_header = ''.join(chr(b) if chr(b) in string.printable else '.' for b in header)
        print("--- Inizio header NetCDF3 (testuale) ---")
        print(ascii_header)
        print("--- ... ---")
    loaded = xr.open_dataarray(out_path)
    assert "wcs_header_dict" in loaded.attrs or loaded.attrs.get("has_wcs", 0) == 1

def test_continuum_removal_save_csv_textual():
    import numpy as np
    import os
    import pandas as pd
    from astrogea.core import continuum_removal
    # Dati sintetici
    img = np.random.rand(3, 4, 8) * 1000
    wavelength = np.linspace(1000, 2000, 8)
    MIN, MAX = 1100, 1900
    result, x = continuum_removal(img, wavelength, MIN, MAX)
    # Prepara DataFrame per CSV: colonne y, x, wavelength, valore
    y_idx, x_idx, w_idx = np.where(~np.isnan(result))
    data = {
        'y': y_idx,
        'x': x_idx,
        'wavelength': x[w_idx],
        'value': result[y_idx, x_idx, w_idx]
    }
    df = pd.DataFrame(data)
    out_path = os.path.abspath("out_continuum_textual.csv")
    df.to_csv(out_path, index=False)
    print(f"File CSV testuale salvato: {out_path}")
    # Stampa le prime righe per verifica
    print("--- Inizio file CSV ---")
    with open(out_path, "r") as f:
        for _ in range(10):
            print(f.readline().rstrip())
    print("--- ... ---")
    assert os.path.exists(out_path)
