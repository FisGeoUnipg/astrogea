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
Convert continuum removal results to an xarray.DataArray with optional WCS coordinates:

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

See the function docstring for details on parameters and output.

### Smoothing Moving Average
The `smoothing_moving_average` function applies a moving average to a hyperspectral cube within the specified wavelength range. It supports both NumPy and Dask arrays for parallel computation.

```python
from astrogea.core import smoothing_moving_average
import numpy as np

# Example data
img = np.random.rand(5, 5, 20) * 1000  # (Y, X, bands)
wavelength = np.linspace(1000, 2500, 20)
MIN, MAX = 1200, 2200
window_size = 5

# Apply smoothing moving average
result, x = smoothing_moving_average(img, wavelength, MIN, MAX, window_size=window_size)
```

**With Dask (parallel computation):**
```python
import dask.array as da
from astrogea.core import smoothing_moving_average

img = da.from_array(np.random.rand(5, 5, 20) * 1000, chunks=(2, 2, 20))
result, x = smoothing_moving_average(img, wavelength, MIN, MAX, window_size=5, use_dask=True)
```

**Returns:**
- `result`: smoothed cube (Y, X, n_bands)
- `x`: corresponding wavelengths

### Spectral Coregistration
The `coregister_spectra` function allows you to realign (coregister) a spectrum onto a reference wavelength grid using linear interpolation. Compatible with both NumPy and Dask arrays as input.

**Example usage:**
```python
from astrogea.core import coregister_spectra
import numpy as np

reference_wavelengths = np.linspace(1000, 2000, 10)
new_wavelengths = np.linspace(950, 2050, 12)
new_reflectance = np.sin(new_wavelengths/300)

# Realign the spectrum onto the reference grid
interpolated = coregister_spectra(reference_wavelengths, new_wavelengths, new_reflectance)
print(interpolated)
```

**Returns:**
- An array of reflectance values interpolated onto the reference_wavelengths grid.

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
    Applies a moving average smoothing to a hyperspectral cube within the specified wavelength range.
    Supports parallel computation with Dask if use_dask=True.

    Parameters
    ----------
    img : np.ndarray or dask.array.Array
        Hyperspectral cube (Y, X, bands)
    wavelength : np.ndarray
        Array of wavelengths (bands,)
    MIN, MAX : float
        Wavelength range limits for analysis
    window_size : int, default 3
        Window size for the moving average
    use_dask : bool, default False
        If True, uses Dask for parallel computation

    Returns
    -------
    result : np.ndarray or dask.array.Array
        Smoothed cube (Y, X, n_bands)
    x : np.ndarray
        Corresponding wavelengths
    """
```

### coregister_spectra
```python
def coregister_spectra(reference_wavelengths, new_wavelengths, new_reflectance):
    """
    Interpolates the reflectance values of a spectrum onto a reference wavelength grid.
    Compatible with both NumPy and Dask arrays as input.

    Parameters
    ----------
    reference_wavelengths : array-like
        Array of target wavelength values (reference grid)
    new_wavelengths : array-like
        Array of original wavelengths of the spectrum to be realigned
    new_reflectance : array-like
        Array of reflectance values corresponding to new_wavelengths

    Returns
    -------
    interpolated_reflectance : np.ndarray
        Reflectance values interpolated onto the reference_wavelengths grid
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
Applies automatic contrast stretching to an RGB image.
```python
from astrogea.core import auto_stretch_rgb
import numpy as np

img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
stretched = auto_stretch_rgb(img)
```

### unison_shuffled_copies
Synchronously shuffles two arrays along the first dimension.
```python
from astrogea.core import unison_shuffled_copies
import numpy as np

a = np.arange(10).reshape(5, 2)
b = np.arange(5)
a_shuf, b_shuf = unison_shuffled_copies(a, b)
```

### merge_datacubes
Merges a list of datacubes along the specified axis (default: 0).
Returns an `xarray.DataArray` with WCS metadata.
```python
from astrogea.core import merge_datacubes
import numpy as np

cube1 = np.random.rand(2, 2, 3)
cube2 = np.random.rand(2, 2, 3)
merged = merge_datacubes([cube1, cube2], axis=0)
```

### spetial_merge_datacubes
Merges two datacubes along the spatial axis (first dimension).
Returns an `xarray.DataArray` with WCS metadata.
```python
from astrogea.core import spetial_merge_datacubes
import numpy as np

cube1 = np.random.rand(2, 2, 3)
cube2 = np.random.rand(3, 2, 3)
merged = spetial_merge_datacubes(cube1, cube2)
```

### hypermerge_spatial
Returns the mean of a list of datacubes (same shape as a single cube), as an `xarray.DataArray` with WCS metadata.
```python
from astrogea.core import hypermerge_spatial
import numpy as np

cube1 = np.random.rand(2, 2, 3)
cube2 = np.random.rand(2, 2, 3)
merged = hypermerge_spatial([cube1, cube2])
```

## xarray Output and WCS Metadata
All main functions return `xarray.DataArray` objects with WCS metadata. If WCS metadata is not available, a dummy WCS header is added to ensure compatibility and interoperability.

## Utility Functions

### auto_stretch_rgb
Applies automatic contrast stretching to an RGB image.
```python
from astrogea.core import auto_stretch_rgb
import numpy as np

img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
stretched = auto_stretch_rgb(img)
```

### unison_shuffled_copies
Synchronously shuffles two arrays along the first dimension.
```python
from astrogea.core import unison_shuffled_copies
import numpy as np

a = np.arange(10).reshape(5, 2)
b = np.arange(5)
a_shuf, b_shuf = unison_shuffled_copies(a, b)
```

### merge_datacubes
Merges a list of datacubes along the specified axis (default: 0).
Returns an `xarray.DataArray` with WCS metadata.
```python
from astrogea.core import merge_datacubes
import numpy as np

cube1 = np.random.rand(2, 2, 3)
cube2 = np.random.rand(2, 2, 3)
merged = merge_datacubes([cube1, cube2], axis=0)
```

### spetial_merge_datacubes
Merges two datacubes along the spatial axis (first dimension).
Returns an `xarray.DataArray` with WCS metadata.
```python
from astrogea.core import spetial_merge_datacubes
import numpy as np

cube1 = np.random.rand(2, 2, 3)
cube2 = np.random.rand(3, 2, 3)
merged = spetial_merge_datacubes(cube1, cube2)
```

### hypermerge_spatial
Returns the mean of a list of datacubes (same shape as a single cube), as an `xarray.DataArray` with WCS metadata.
```python
from astrogea.core import hypermerge_spatial
import numpy as np

cube1 = np.random.rand(2, 2, 3)
cube2 = np.random.rand(2, 2, 3)
merged = hypermerge_spatial([cube1, cube2])
```
