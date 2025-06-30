import numpy as np
import spectral.io.envi as envi
import xarray as xr
import time
import os
import warnings
from typing import Optional, Dict, Any
from .spectral_wrapper import SpectralArrayWrapper
from .wcs_utils import parse_envi_map_info_list, create_wcs_from_parsed_info, create_wcs_header_dict

def _setup_spectral_environment():
    """Setup spectral library environment to find data files"""
    # Add current directory and data directory to spectral search path
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, "data")
    
    # Set SPECTRAL_DATA environment variable if not already set
    if "SPECTRAL_DATA" not in os.environ:
        spectral_paths = [current_dir, data_dir]
        os.environ["SPECTRAL_DATA"] = os.pathsep.join(spectral_paths)
    else:
        # Add our paths to existing SPECTRAL_DATA
        existing_paths = os.environ["SPECTRAL_DATA"].split(os.pathsep)
        if current_dir not in existing_paths:
            existing_paths.append(current_dir)
        if data_dir not in existing_paths:
            existing_paths.append(data_dir)
        os.environ["SPECTRAL_DATA"] = os.pathsep.join(existing_paths)

def _open_envi_file(base_path: str) -> Any:
    """Open ENVI file with proper path handling"""
    # Setup spectral environment
    _setup_spectral_environment()
    
    # Remove .hdr extension if present to get base path
    if base_path.endswith('.hdr'):
        base_path = base_path[:-4]
    
    # Try different path combinations
    hdr_path = f"{base_path}.hdr"
    img_path = f"{base_path}.img"
    
    # Check if files exist
    if not os.path.exists(hdr_path):
        raise FileNotFoundError(f"Header file not found: {hdr_path}")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")
    
    # Use absolute paths
    hdr_abs = os.path.abspath(hdr_path)
    img_abs = os.path.abspath(img_path)
    
    return envi.open(hdr_abs, img_abs)

def process_crism_file(base_sr_path: str, base_if_path: str, output_nc_path: str, use_dask: bool = False) -> xr.Dataset:
    """
    Process CRISM files and save them in NetCDF format with georeferencing.
    Matches the functionality of the reference script.
    
    Args:
        base_sr_path: Base path to the .hdr Surface Reflectance file (without extension)
        base_if_path: Base path to the .hdr Incidence Factor file (without extension)
        output_nc_path: Output NetCDF file
        use_dask: If True, use Dask to load files for better performance with large datasets
    
    Returns:
        xarray.Dataset: Processed dataset with georeferenced data
    """
    # Load SR (surface reflectance)
    sr_img = _open_envi_file(base_sr_path)
    
    if use_dask:
        try:
            import dask.array as da
            from dask.diagnostics import ProgressBar
            
            # Load with Dask for better performance
            sr_data = sr_img.load()
            sr_data_backend = da.from_array(sr_data, chunks=('auto', 'auto', -1))
            print("   Using Dask backend for SR data")
        except ImportError:
            print("   Dask not available, falling back to NumPy")
            sr_data = sr_img.load()
            sr_data_backend = sr_data
    else:
        sr_data = sr_img.load()
        sr_data_backend = sr_data
    
    # Extract wavelengths from SR
    wavelengths = np.linspace(1.0, 2.5, sr_data.shape[2])  # Default wavelengths
    wavelength_units = 'micrometers'
    if hasattr(sr_img, 'bands') and sr_img.bands.centers:
        wavelengths = np.array(sr_img.bands.centers, dtype=np.float32)
        wavelength_units = sr_img.metadata.get('wavelength units', 'unknown')
    
    # Load IF (incidence factor)
    if_img = _open_envi_file(base_if_path)
    
    if use_dask:
        try:
            import dask.array as da
            # Load with Dask for better performance
            if_data = if_img.load()
            if_data_backend = da.from_array(if_data, chunks=('auto', 'auto', -1))
            print("   Using Dask backend for IF data")
        except ImportError:
            print("   Dask not available, falling back to NumPy")
            if_data = if_img.load()
            if_data_backend = if_data
    else:
        if_data = if_img.load()
        if_data_backend = if_data
    
    # Extract IF band names
    if_band_names = [f"if_band_{i}" for i in range(if_data.shape[2])]
    if hasattr(if_img, 'metadata') and 'band names' in if_img.metadata:
        bn = if_img.metadata['band names']
        if isinstance(bn, list) and len(bn) == if_data.shape[2]:
            import re
            if_band_names = [re.sub(r'\s*\(.*\)\s*$', '', b).strip() for b in bn]
    
    # Extract metadata
    meta_sr = {k: str(v) for k, v in sr_img.metadata.items()} if hasattr(sr_img, 'metadata') else {}
    meta_if = {k: str(v) for k, v in if_img.metadata.items()} if hasattr(if_img, 'metadata') else {}
    
    # Create WCS from SR metadata
    wcs_object = None
    wcs_header_dict = None
    map_info_data = sr_img.metadata.get('map info', None)
    parsed_map_info = None
    
    if map_info_data and isinstance(map_info_data, list):
        parsed_map_info = parse_envi_map_info_list(map_info_data)
        if parsed_map_info:
            wcs_object = create_wcs_from_parsed_info(parsed_map_info, sr_data.shape[:2])
            if wcs_object:
                wcs_header_dict = create_wcs_header_dict(wcs_object, parsed_map_info)
    
    # Create coordinates
    lines, samples = sr_data.shape[:2]
    line_coords = xr.DataArray(
        np.arange(lines), 
        dims='line', 
        attrs={'long_name': 'Spatial dimension Y', 'units': 'pixel_index'}
    )
    sample_coords = xr.DataArray(
        np.arange(samples), 
        dims='sample', 
        attrs={'long_name': 'Spatial dimension X', 'units': 'pixel_index'}
    )
    wavelength_coords = xr.DataArray(
        wavelengths, 
        dims='wavelength', 
        attrs={'long_name': 'Wavelength', 'units': wavelength_units}
    )
    if_band_coords = xr.DataArray(
        if_band_names, 
        dims='if_band', 
        attrs={'long_name': 'Geometry Band'}
    )
    
    # Create DataArrays
    sr_da = xr.DataArray(
        data=sr_data_backend,
        coords={'line': line_coords, 'sample': sample_coords, 'wavelength': wavelength_coords},
        dims=['line', 'sample', 'wavelength'],
        name='spectral_data',
        attrs={
            'description': meta_sr.get('description', 'Spectral data'),
            'source_file': f"{base_sr_path}.hdr"
        }
    )
    
    if_da = xr.DataArray(
        data=if_data_backend,
        coords={'line': line_coords, 'sample': sample_coords, 'if_band': if_band_coords},
        dims=['line', 'sample', 'if_band'],
        name='geometry_angles',
        attrs={
            'description': meta_if.get('description', 'Geometry data'),
            'source_file': f"{base_if_path}.hdr"
        }
    )
    
    # Create global attributes
    global_attrs = {
        'title': f'CRISM Combined Dataset for {os.path.basename(base_sr_path)}',
        'source_files_base': f"{base_sr_path}, {base_if_path}",
        'processing_script': 'astrogea library',
        'creation_date': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        'Conventions': 'CF-1.8',
        'history': f'{time.strftime("%Y-%m-%d %H:%M:%S")}: Created dataset using astrogea library.'
    }
    
    # Add WCS information
    if wcs_header_dict:
        global_attrs['wcs_header_dict'] = str(wcs_header_dict)
        global_attrs['has_wcs'] = 1
        global_attrs['has_wcs_comment'] = "Integer flag: 1 = WCS header present, 0 = no WCS info."
    else:
        global_attrs['has_wcs'] = 0
        global_attrs['has_wcs_comment'] = "Integer flag: 1 = WCS header present, 0 = no WCS info."
    
    # Create dataset
    ds = xr.Dataset(
        {
            'spectral_data': sr_da,
            'geometry_angles': if_da
        },
        coords={
            'line': line_coords,
            'sample': sample_coords,
            'wavelength': wavelength_coords,
            'if_band': if_band_coords
        },
        attrs=global_attrs
    )
    
    # Save in NetCDF
    encoding_options = {}
    
    # Only add compression if netCDF4 is available
    try:
        import netCDF4
        encoding_options = {
            'spectral_data': {'zlib': True, 'complevel': 4},
            'geometry_angles': {'zlib': True, 'complevel': 4}
        }
        
        if sr_da.dtype.kind == 'f':
            encoding_options['spectral_data']['_FillValue'] = np.nan
        if if_da.dtype.kind == 'f':
            encoding_options['geometry_angles']['_FillValue'] = np.nan
    except ImportError:
        # Use scipy backend without compression
        pass
    
    if use_dask:
        try:
            from dask.diagnostics import ProgressBar
            print("   Saving with Dask backend...")
            with ProgressBar():
                ds.to_netcdf(output_nc_path, encoding=encoding_options)
        except ImportError:
            ds.to_netcdf(output_nc_path, encoding=encoding_options)
    else:
        ds.to_netcdf(output_nc_path, encoding=encoding_options)
    
    return ds

def envi_to_xarray_wcs(hdr_path: str, img_path: str = None) -> xr.Dataset:
    """
    Convert an ENVI file (.hdr/.img) to an xarray with WCS coordinates using only astrogea.
    
    Args:
        hdr_path: Path to the .hdr file
        img_path: Path to the .img file (if None, uses the same name as hdr_path)
    
    Returns:
        xarray.Dataset: Dataset with WCS coordinates
    """
    if img_path is None:
        img_path = hdr_path
    
    # Load the ENVI file using spectral
    img = _open_envi_file(hdr_path)
    data = img.load()
    
    # Extract wavelengths
    wavelengths = np.linspace(1.0, 2.5, data.shape[2])  # Default wavelengths
    wavelength_units = 'micrometers'
    if hasattr(img, 'bands') and img.bands.centers:
        wavelengths = np.array(img.bands.centers, dtype=np.float32)
        wavelength_units = img.metadata.get('wavelength units', 'unknown')
    
    # Use astrogea's SpectralArrayWrapper to convert to xarray
    wrapper = SpectralArrayWrapper(data, wavelengths)
    xarray_data = wrapper.to_xarray()
    
    # Create coordinates
    lines, samples = data.shape[:2]
    line_coords = xr.DataArray(
        np.arange(lines), 
        dims='line', 
        attrs={'long_name': 'Spatial dimension Y', 'units': 'pixel_index'}
    )
    sample_coords = xr.DataArray(
        np.arange(samples), 
        dims='sample', 
        attrs={'long_name': 'Spatial dimension X', 'units': 'pixel_index'}
    )
    wavelength_coords = xr.DataArray(
        wavelengths, 
        dims='wavelength', 
        attrs={'long_name': 'Wavelength', 'units': wavelength_units}
    )
    
    # Create base dataset
    ds = xr.Dataset({
        "spectral_data": xarray_data
    })
    
    # Apply georeferencing if available
    map_info = img.metadata.get("map info")
    wcs_object = None
    wcs_header_dict = None
    
    if map_info:
        parsed_info = parse_envi_map_info_list(map_info)
        if parsed_info:
            wcs_object = create_wcs_from_parsed_info(parsed_info, data.shape[:2])
            if wcs_object:
                wcs_header_dict = create_wcs_header_dict(wcs_object, parsed_info)
    
    # Create global attributes
    global_attrs = {
        'title': f'ENVI Dataset for {os.path.basename(hdr_path)}',
        'source_file': hdr_path,
        'processing_script': 'astrogea library',
        'creation_date': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        'Conventions': 'CF-1.8',
        'history': f'{time.strftime("%Y-%m-%d %H:%M:%S")}: Created dataset using astrogea library.'
    }
    
    # Add WCS information
    if wcs_header_dict:
        global_attrs['wcs_header_dict'] = str(wcs_header_dict)
        global_attrs['has_wcs'] = 1
        global_attrs['has_wcs_comment'] = "Integer flag: 1 = WCS header present, 0 = no WCS info."
    else:
        global_attrs['has_wcs'] = 0
        global_attrs['has_wcs_comment'] = "Integer flag: 1 = WCS header present, 0 = no WCS info."
    
    ds.attrs.update(global_attrs)
    
    return ds

def continuum_removal(img, wavelength, MIN, MAX, interp='linear', force=False, forcemin=1500, forcemax=1800, use_dask=False):
    """
    Continuum removal usando interpolazione hull convesso per ogni pixel di un cubo iperspettrale.
    Supporta calcolo parallelo con Dask se use_dask=True.

    Parameters
    ----------
    img : np.ndarray or dask.array.Array
        Cubo iperspettrale (I, J, K)
    wavelength : np.ndarray
        Array delle lunghezze d'onda (K,)
    MIN, MAX : float
        Limiti di lunghezza d'onda per l'analisi
    interp : str, default 'linear'
        Tipo di interpolazione ('linear', 'cubic', ...)
    force : bool, default False
        Se True forza il passaggio del continuo per un punto specifico
    forcemin, forcemax : float, default 1500, 1800
        Range di lunghezze d'onda per il punto forzato
    use_dask : bool, default False
        Se True usa Dask per il calcolo parallelo

    Returns
    -------
    result : np.ndarray or dask.array.Array
        Cubo normalizzato (I, J, n_bands)
    x : np.ndarray
        Lunghezze d'onda corrispondenti
    """
    import numpy as np
    from scipy.interpolate import interp1d
    from scipy.spatial import ConvexHull

    # Dask support
    if use_dask:
        try:
            import dask.array as da
        except ImportError:
            warnings.warn("Dask non disponibile, uso NumPy.")
            da = None
            use_dask = False
    else:
        da = None

    I, J, K = img.shape
    limI = int(np.argmin(np.abs(wavelength - MIN)))
    limU = int(np.argmin(np.abs(wavelength - MAX)))
    n = limU - limI

    if force:
        limA = int(np.argmin(np.abs(wavelength - forcemin)))
        limB = int(np.argmin(np.abs(wavelength - forcemax)))

    if n < 2:
        raise ValueError("Too few bands in selected range.")

    x = wavelength[limI:limU]

    def _process_pixel(spectrum):
        # skip invalid values (marked in CRISM hyperspectral data as 65535)
        if spectrum[0] == 65535:
            return np.full((n,), np.nan)
        y = spectrum[limI:limU]
        points = np.c_[x, y]
        # process to build the convex hull
        augmented = np.concatenate([points, [(x[0], np.min(y)-1), (x[-1], np.min(y)-1)]], axis=0)
        hull = ConvexHull(augmented, incremental=True)
        continuum_points = points[np.sort([v for v in hull.vertices if v < len(points)])]
        # forcing the hull to pass through the point
        if force:
            sub_x = wavelength[limA:limB]
            sub_y = spectrum[limA:limB]
            if len(sub_y) == 0:
                return np.full((n,), np.nan)
            max_idx = np.argmax(sub_y)
            max_wave = sub_x[max_idx]
            max_val = sub_y[max_idx]
            max_point = np.array([[max_wave, max_val]])
            continuum_points = np.vstack([continuum_points, max_point])
            continuum_points = continuum_points[np.argsort(continuum_points[:, 0])]
        # interpolate the convex hull to have the same length as the spectrum
        continuum_function = interp1d(*continuum_points.T, kind=interp, fill_value="extrapolate")
        neutral = continuum_function(x)
        # compute the "corrected" spectrum as a ratio between the spectrum and the hull
        return y / neutral

    if use_dask and da is not None and isinstance(img, da.Array):
        # Dask parallelization: map_blocks expects (I, J, K) -> (I, J, n)
        def _dask_apply(block):
            out = np.empty((block.shape[0], block.shape[1], n), dtype=block.dtype)
            for i in range(block.shape[0]):
                for j in range(block.shape[1]):
                    out[i, j, :] = _process_pixel(block[i, j, :])
            return out
        result = img.map_blocks(_dask_apply, dtype=img.dtype, chunks=(img.chunks[0], img.chunks[1], n))
    else:
        result = np.zeros((I, J, n), dtype=img.dtype)
        for i in range(I):
            for j in range(J):
                result[i, j, :] = _process_pixel(img[i, j, :])

    return result, x

def continuum_to_xarray_wcs(result, x, parsed_map_info=None):
    """
    Utility: crea un xarray.Dataset dal risultato di continuum_removal, con coordinate e attributi WCS opzionali.
    result: array (Y, X, bands)
    x: array delle lunghezze d'onda
    parsed_map_info: dict opzionale, come restituito da parse_envi_map_info_list
    """
    import xarray as xr
    import numpy as np
    from .wcs_utils import create_wcs_from_parsed_info, create_wcs_header_dict
    y, x_dim, bands = result.shape
    coords = {
        'y': np.arange(y),
        'x': np.arange(x_dim),
        'wavelength': x
    }
    da = xr.DataArray(
        data=result,
        dims=["y", "x", "wavelength"],
        coords=coords,
        name="continuum_removed"
    )
    attrs = {}
    if parsed_map_info is not None:
        wcs_obj = create_wcs_from_parsed_info(parsed_map_info, (y, x_dim))
        if wcs_obj is not None:
            wcs_header = create_wcs_header_dict(wcs_obj, parsed_map_info)
            attrs['wcs_header_dict'] = str(wcs_header)
            attrs['has_wcs'] = 1
        else:
            attrs['has_wcs'] = 0
    da.attrs.update(attrs)
    return da

def band_parameters_mafic(img_removed, wav, nbands=5, windows_nm=75, resolution_nm=5, tol=10, use_dask=False):
    """
    Calcola i parametri delle bande (minimo, centro, profondità, area, asimmetria) su uno spettro continuum-removed.
    Supporta calcolo parallelo con Dask se use_dask=True.

    Parameters
    ----------
    img_removed : np.ndarray or dask.array.Array
        Cubo continuum-removed (I, J, K)
    wav : np.ndarray
        Array delle lunghezze d'onda (K,)
    nbands : int, default 5
        Numero massimo di bande da cercare
    windows_nm : float, default 75
        Finestra in nm per la ricerca del centro banda
    resolution_nm : float, default 5
        Risoluzione in nm per l'interpolazione
    tol : int, default 10
        Soglia minima di ampiezza banda (in pixel)
    use_dask : bool, default False
        Se True usa Dask per il calcolo parallelo

    Returns
    -------
    mappa : np.ndarray or dask.array.Array
        Array (I, J, nbands*5) con i parametri delle bande
    """
    import numpy as np
    from scipy.interpolate import interp1d
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    I, J, K = img_removed.shape
    mappa = np.zeros((I, J, int(nbands*5)))
    x = wav

    def _process_pixel(y):
        if y[0] == 0:
            return np.zeros(nbands*5)
        ones_indexes = np.argwhere(y == 1)
        ones_idx = [ones_indexes[i][0] for i in range(len(ones_indexes))]
        l = 0
        out = np.zeros(nbands*5)
        for i in range(0, len(ones_idx) - 1):
            if ones_idx[i+1] - ones_idx[i] >= tol:
                S = y[ones_idx[i]:ones_idx[i+1]]
                X = x[ones_idx[i]:ones_idx[i+1]]
                if len(S) < 3:
                    continue
                minimum, minimum_index = np.min(S), np.argmin(S)
                shift_right = find_nearest(X, X[minimum_index]+windows_nm)
                shift_left = find_nearest(X, X[minimum_index]-windows_nm)
                Xfit_range = np.arange(X[shift_left], X[shift_right]+resolution_nm, resolution_nm)
                interp_S = interp1d(X, S, kind='cubic', fill_value="extrapolate")(Xfit_range)
                coeffs = np.polyfit(Xfit_range, interp_S, 4)
                poly = np.poly1d(coeffs)
                y_poly = poly(Xfit_range)
                idx_center = np.argmin(y_poly)
                band_center_wav = Xfit_range[idx_center]
                band_center_val = y_poly[idx_center]
                band_depth = 1 - S[idx_center] if idx_center < len(S) else 0
                total_area = np.trapz(np.ones_like(S) - S, X)
                left_area = np.trapz(S[:idx_center], X[:idx_center]) if idx_center > 0 else 0
                right_area = np.trapz(S[idx_center:], X[idx_center:]) if idx_center < len(S) else 0
                asymmetry = (right_area - left_area) / (100 * total_area) if total_area != 0 else 0
                out[l] = X[minimum_index]
                out[l+1] = band_center_wav
                out[l+2] = band_depth
                out[l+3] = total_area
                out[l+4] = asymmetry
                l += 5
                if l >= nbands*5:
                    break
        return out

    # Dask support
    if use_dask:
        try:
            import dask.array as da
        except ImportError:
            da = None
            use_dask = False
    else:
        da = None

    if use_dask and da is not None and isinstance(img_removed, da.Array):
        def _dask_apply(block):
            out = np.zeros((block.shape[0], block.shape[1], nbands*5), dtype=block.dtype)
            for i in range(block.shape[0]):
                for j in range(block.shape[1]):
                    out[i, j, :] = _process_pixel(block[i, j, :])
            return out
        result = img_removed.map_blocks(_dask_apply, dtype=img_removed.dtype, chunks=(img_removed.chunks[0], img_removed.chunks[1], nbands*5))
        return result
    else:
        for I in range(I):
            for J in range(J):
                mappa[I, J, :] = _process_pixel(img_removed[I, J, :])
        return mappa
