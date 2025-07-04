# astrogea

AstroGea is a Python-based framework designed to bridge the realms of astrophysics and geoscience through advanced machine learning and intuitive data visualization. It serves as a unified platform for researchers and analysts to explore, model, and visualize complex datasets from both celestial and terrestrial sources.

🔭 Features
- Unified Data Handling: Seamlessly integrate and process datasets from astronomical observations and geological surveys.
- Machine Learning Integration: Apply state-of-the-art ML algorithms to uncover patterns and insights across diverse scientific domains.
- Interactive Visualizations: Generate dynamic, interactive visualizations that elucidate complex relationships within the data.
- Modular Architecture: Flexible design allows for easy extension and customization to suit specific research needs.
- Spectral Analysis: Advanced spectral processing tools including continuum removal, hyperspectral data analysis, moving average smoothing, and spectral coregistration (NumPy & Dask support).

🌐 Use Cases
<!-- - Astrogeological Analysis: Study the interplay between cosmic events and geological phenomena. -->
- Planetary Science: Analyze planetary compositions and structures using integrated datasets.
- Educational Tools: Develop interactive modules for teaching concepts in astrophysics and geology.
- Spectral Analysis: Process and analyze hyperspectral data from various sources.

## Core Features

### Continuum Removal
The `continuum_removal` function performs convex hull continuum removal on hyperspectral cubes. It supports both NumPy and Dask for parallel processing.

```python
from astrogea.core import continuum_removal
import numpy as np

# Create sample data
img = np.random.rand(5, 5, 20) * 1000  # (Y, X, bands)
wavelength = np.linspace(1000, 2500, 20)
MIN, MAX = 1200, 2200

# Perform continuum removal
result, x = continuum_removal(img, wavelength, MIN, MAX)
```

**With Dask (parallel processing):**
```python
import dask.array as da
from astrogea.core import continuum_removal

# Create Dask array
img = da.from_array(np.random.rand(5, 5, 20) * 1000, chunks=(2, 2, 20))
result, x = continuum_removal(img, wavelength, MIN, MAX, use_dask=True)
```

### WCS Integration
Convert continuum removal results to xarray.DataArray with optional WCS coordinates:

```python
from astrogea.core import continuum_to_xarray_wcs
from astrogea.wcs_utils import parse_envi_map_info_list

# Create WCS information
map_info = ["UTM", 1.0, 2.0, 100.0, 200.0, 30.0, 30.0, "meters"]
parsed_map_info = parse_envi_map_info_list(map_info)

# Create xarray DataArray with WCS
da = continuum_to_xarray_wcs(result, x, parsed_map_info)
```

### Data Export Options
The processed data can be saved in various formats:

```python
# NetCDF4 (binary with WCS)
da.to_netcdf("out_continuum.nc", engine="netcdf4")

# NetCDF3 (text header)
da.to_netcdf("out_continuum_textual.nc", engine="scipy", format="NETCDF3_CLASSIC")

# CSV (fully textual)
import pandas as pd
y_idx, x_idx, w_idx = np.where(~np.isnan(result))
df = pd.DataFrame({
    'y': y_idx,
    'x': x_idx,
    'wavelength': x[w_idx],
    'value': result[y_idx, x_idx, w_idx]
})
df.to_csv("out_continuum_textual.csv", index=False)
```

### Mafic Band Analysis
The `band_parameters_mafic` function extracts mafic band parameters (minimum, center, depth, area, asymmetry) from continuum-removed hyperspectral data. It supports both NumPy and Dask arrays for parallel processing.

**Example usage:**
```python
from astrogea.core import continuum_removal, band_parameters_mafic
import numpy as np

# Synthetic data: 4x4x10 cube
img = np.random.rand(4, 4, 10) * 1000
wavelength = np.linspace(1000, 2000, 10)
MIN, MAX = 1100, 1900

# Continuum removal
result, x = continuum_removal(img, wavelength, MIN, MAX)
# Mafic band analysis
mafic_map = band_parameters_mafic(result, x, nbands=3)
print("Mafic band parameters (NumPy):\n", mafic_map)
```

**With Dask (parallel processing):**
```python
import dask.array as da
from astrogea.core import continuum_removal, band_parameters_mafic

img = da.from_array(np.random.rand(4, 4, 10) * 1000, chunks=(2, 2, 10))
wavelength = np.linspace(1000, 2000, 10)
result, x = continuum_removal(img, wavelength, MIN, MAX, use_dask=True)
# Compute result if needed, then analyze
mafic_map = band_parameters_mafic(result.compute(), x, nbands=3)
print("Mafic band parameters (Dask, computed):\n", mafic_map)
# Or analyze directly with Dask (fully parallel)
mafic_map_dask = band_parameters_mafic(img, wavelength, nbands=3, use_dask=True)
print("Mafic band parameters (Dask, parallel):\n", mafic_map_dask.compute())
```

**Returns:**
- `mafic_map`: array (Y, X, nbands*5) with band parameters for each pixel.

See function docstring for details on parameters and output.

### Smoothing Moving Average
La funzione `smoothing_moving_average` applica una media mobile (moving average) su un cubo iperspettrale nell'intervallo di lunghezze d'onda specificato. Supporta sia array NumPy che Dask per il calcolo parallelo.

```python
from astrogea.core import smoothing_moving_average
import numpy as np

# Dati di esempio
img = np.random.rand(5, 5, 20) * 1000  # (Y, X, bands)
wavelength = np.linspace(1000, 2500, 20)
MIN, MAX = 1200, 2200
window_size = 5

# Applica smoothing moving average
result, x = smoothing_moving_average(img, wavelength, MIN, MAX, window_size=window_size)
```

**Con Dask (calcolo parallelo):**
```python
import dask.array as da
from astrogea.core import smoothing_moving_average

img = da.from_array(np.random.rand(5, 5, 20) * 1000, chunks=(2, 2, 20))
result, x = smoothing_moving_average(img, wavelength, MIN, MAX, window_size=5, use_dask=True)
```

**Restituisce:**
- `result`: cubo smoothato (Y, X, n_bands)
- `x`: lunghezze d'onda corrispondenti

### Coregistrazione Spettrale
La funzione `coregister_spectra` permette di riallineare (coregistrare) uno spettro su una griglia di lunghezze d'onda di riferimento tramite interpolazione lineare. Compatibile sia con array NumPy che Dask.

**Esempio d'uso:**
```python
from astrogea.core import coregister_spectra
import numpy as np

reference_wavelengths = np.linspace(1000, 2000, 10)
new_wavelengths = np.linspace(950, 2050, 12)
new_reflectance = np.sin(new_wavelengths/300)

# Riallinea lo spettro sulla griglia di riferimento
interpolated = coregister_spectra(reference_wavelengths, new_wavelengths, new_reflectance)
print(interpolated)
```

**Restituisce:**
- Un array di riflettanza interpolato sulla griglia di riferimento.

## Function Details

### continuum_removal
```python
def continuum_removal(img, wavelength, MIN, MAX, interp='linear', force=False, 
                     forcemin=1500, forcemax=1800, use_dask=False):
    """
    Perform convex hull continuum removal on a hyperspectral cube.
    
    Parameters
    ----------
    img : np.ndarray or dask.array.Array
        Hyperspectral cube (Y, X, bands)
    wavelength : np.ndarray
        Array of wavelengths (bands,)
    MIN, MAX : float
        Wavelength range limits for analysis
    interp : str, default 'linear'
        Interpolation type ('linear', 'cubic', ...)
    force : bool, default False
        If True, forces continuum through a specific point
    forcemin, forcemax : float, default 1500, 1800
        Wavelength range for forced point
    use_dask : bool, default False
        If True, uses Dask for parallel computation
    
    Returns
    -------
    result : np.ndarray or dask.array.Array
        Normalized cube (Y, X, n_bands)
    x : np.ndarray
        Corresponding wavelengths
    """
```

### smoothing_moving_average
```python
def smoothing_moving_average(img, wavelength, MIN, MAX, window_size=3, use_dask=False):
    """
    Applica uno smoothing moving average (media mobile) su un cubo iperspettrale nell'intervallo di lunghezze d'onda specificato.
    Supporta calcolo parallelo con Dask se use_dask=True.

    Parameters
    ----------
    img : np.ndarray or dask.array.Array
        Cubo iperspettrale (Y, X, bands)
    wavelength : np.ndarray
        Array delle lunghezze d'onda (bands,)
    MIN, MAX : float
        Limiti di lunghezza d'onda per l'analisi
    window_size : int, default 3
        Dimensione della finestra per la media mobile
    use_dask : bool, default False
        Se True usa Dask per il calcolo parallelo

    Returns
    -------
    result : np.ndarray or dask.array.Array
        Cubo smoothato (Y, X, n_bands)
    x : np.ndarray
        Lunghezze d'onda corrispondenti
    """
```

### coregister_spectra
```python
def coregister_spectra(reference_wavelengths, new_wavelengths, new_reflectance):
    """
    Interpola i valori di riflettanza di uno spettro su una griglia di lunghezze d'onda di riferimento.
    Compatibile sia con array NumPy che Dask come input.

    Parameters
    ----------
    reference_wavelengths : array-like
        Array dei valori di lunghezza d'onda target (griglia di riferimento)
    new_wavelengths : array-like
        Array delle lunghezze d'onda originali dello spettro da riallineare
    new_reflectance : array-like
        Array dei valori di riflettanza corrispondenti a new_wavelengths

    Returns
    -------
    interpolated_reflectance : np.ndarray
        Valori di riflettanza interpolati sulla griglia reference_wavelengths
    """
```

For more details, refer to the function documentation in `astrogea.core`.

## Additional Spectral Utilities

### remove_crism_bad_ranges_cube
Removes problematic CRISM spectral windows from a hyperspectral datacube.
```python
from astrogea.core import remove_crism_bad_ranges_cube
cube_good, wavelengths_good = remove_crism_bad_ranges_cube(cube, wavelengths_nm)
```

### row_norm, column_norm, center_norm, L1_norm, minmax, robust_scaler, derivative, log_1_r_norm
Various normalization and scaling utilities for spectral data. All support NumPy and Dask arrays via the 'use_dask' parameter.
```python
from astrogea.core import row_norm, column_norm, center_norm, L1_norm, minmax, robust_scaler, derivative, log_1_r_norm
# Example: Normalize spectra using L1 norm
normed = L1_norm(spectra, use_dask=False)
# With Dask:
import dask.array as da
spectra_dask = da.from_array(spectra, chunks=(100, 10))
normed_dask = L1_norm(spectra_dask, use_dask=True)
```

### baseline_correction_cube
Applies baseline correction to a hyperspectral datacube using polynomial fitting.
```python
from astrogea.core import baseline_correction_cube
corrected = baseline_correction_cube(cube, wavelengths_nm, order=1)
```

### auto_stretch_rgb
Automatically stretches each band of a hyperspectral image for RGB visualization.
```python
from astrogea.core import auto_stretch_rgb
stretched = auto_stretch_rgb(img_sr, product_names)
```

### unison_shuffled_copies
Shuffles two arrays in unison using a fixed seed.
```python
from astrogea.core import unison_shuffled_copies
shuffled_spectra, shuffled_indexes = unison_shuffled_copies(spectra, indexes)
```

### merge_datacubes, spetial_merge_datacubes, hypermerge_spatial
Utilities for merging multiple hyperspectral datacubes along different axes.
```python
from astrogea.core import merge_datacubes, spetial_merge_datacubes, hypermerge_spatial
merged = merge_datacubes([cube1, cube2], axis=2)
spatial_merged = spetial_merge_datacubes([cube1, cube2])
hyper_merged = hypermerge_spatial([cube1, cube2])
```
