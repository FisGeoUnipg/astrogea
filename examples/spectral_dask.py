# -*- coding: utf-8 -*-
"""
Script to read a CRISM spectral cube (ENVI-like format),
convert it to an xarray.DataArray (potentially using Dask for
memory efficiency and parallelism) and save it in NetCDF format.

Version with Wrapper class for Dask/Xarray compatibility.
"""

# --- Imports ---
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import warnings # To handle specific warnings
import traceback # For more detailed debugging

# Import from 'spectral' library
try:
    import spectral
    from spectral import open_image
    print(f"Spectralpy library (version: {spectral.__version__}) successfully imported.")
except ImportError:
    print("\nCRITICAL ERROR: 'spectral' library not found.")
    print("Installation instructions: python -m pip install spectralpy")
    exit()
except Exception as e:
    print(f"Unexpected error during spectral import: {e}")
    exit()

# Import other required libraries
try:
    import xarray as xr
    import numpy as np
    import matplotlib.pyplot as plt
    print(f"Other required libraries (xarray {xr.__version__}, numpy {np.__version__}, matplotlib) loaded.")
except ImportError as e:
    print(f"\nCRITICAL ERROR: Missing dependent library: {e}")
    exit()

# Import NetCDF engine
try:
    import netCDF4
    print(f"NetCDF engine (netCDF4) successfully imported.")
    NETCDF_ENGINE = 'netcdf4'
except ImportError:
    print("\nWARNING: 'netCDF4' library not found. Install with: python -m pip install netCDF4")
    NETCDF_ENGINE = None

# --- Dask Import ---
try:
    import dask.array as da
    from dask.diagnostics import ProgressBar
    print(f"Dask library successfully imported.")
    USE_DASK = True
except ImportError:
    print("\nWARNING: 'dask' library not found. Install with: python -m pip install \"dask[complete]\"")
    USE_DASK = False

# --- Spectral Object Wrapper Class Definition ---
class SpectralArrayWrapper:
    """
    A wrapper to make the spectral.Image object (like BsqFile)
    more compatible with interfaces expecting ndim.
    Delegates shape, dtype and __getitem__ to the underlying object.
    """
    def __init__(self, spectral_image):
        self._spectral_image = spectral_image
        self.shape = spectral_image.shape
        self.dtype = spectral_image.dtype
        # Add missing 'ndim' attribute
        self.ndim = len(self.shape)
        print(f"  [Wrapper] Created wrapper: shape={self.shape}, dtype={self.dtype}, ndim={self.ndim}")

    def __getitem__(self, key):
        # Delegate indexing/slicing to original object
        # print(f"  [Wrapper] __getitem__ called with key: {key}") # Debug (can be noisy)
        return self._spectral_image[key]

# --- Configuration ---
# !!! MODIFY HERE the path to your .hdr file !!!
hdr_file_path = 'frt000144ff_07_sr164j_mtr3.hdr'

base_name = os.path.splitext(os.path.basename(hdr_file_path))[0]
output_nc_file = f"{base_name}_xarray_dask.nc" if USE_DASK else f"{base_name}_xarray.nc"
print(f"\nInput Header file: '{hdr_file_path}'")
print(f"Planned NetCDF output file: '{output_nc_file}'")

CHUNKS = {'line': 'auto', 'sample': 'auto', 'wavelength': -1}
print(f"Dask Chunks configuration (if used): {CHUNKS}")

# --- Check Header File Existence ---
print(f"\nChecking header file existence: '{hdr_file_path}'")
if not os.path.exists(hdr_file_path):
    print(f"--> ERROR: Specified header file NOT found: {os.path.abspath(hdr_file_path)}")
    exit()
else:
    print("--> OK: Header file found.")

# --- Spectral Loading ---
img_spectral = None
try:
    print(f"\nAttempting to open image file via header: {hdr_file_path}")
    img_spectral = open_image(hdr_file_path)
    print("--> OK: Image object successfully opened using spectral.")
    print(f"  Type: {type(img_spectral)}, Shape: {img_spectral.shape}, Dtype: {img_spectral.dtype}")
    if not hasattr(img_spectral, 'metadata'):
        print("  Warning: Image object does not have 'metadata' attribute.")

except FileNotFoundError:
    img_file_expected = hdr_file_path.replace('.hdr', '.img')
    if not os.path.exists(img_file_expected): img_file_expected = hdr_file_path.replace('.hdr', '') # Try without extension
    if not os.path.exists(img_file_expected):
         print(f"--> ERROR: Unable to find image file (.img or without extension) associated with '{hdr_file_path}'")
    else:
         print(f"--> ERROR: Unexplained FileNotFoundError during opening.")
    exit()
except Exception as e:
    print(f"\n--> ERROR during file opening with spectral: {type(e).__name__}: {e}")
    traceback.print_exc()
    exit()

# --- Metadata Extraction ---
lines, samples, bands = 0, 0, 0
wavelengths = None
wavelength_units = 'unknown'
data_description = 'Spectral Data Cube'
original_metadata = {}

try:
    print("\nExtracting metadata from spectral object...")
    if not hasattr(img_spectral, 'shape') or not hasattr(img_spectral, 'dtype'):
        print("--> ERROR: Spectral object does not have 'shape' or 'dtype' attributes.")
        exit()

    lines, samples, bands = img_spectral.shape
    print(f"  Dimensions: Lines={lines}, Samples={samples}, Bands={bands}")

    print("  Extracting wavelengths...")
    source = "N/A"
    if hasattr(img_spectral, 'bands') and hasattr(img_spectral.bands, 'centers') and img_spectral.bands.centers:
        wavelengths = np.array(img_spectral.bands.centers, dtype=np.float32)
        source = "'img_spectral.bands.centers'"
    elif hasattr(img_spectral, 'metadata') and 'wavelength' in img_spectral.metadata:
        wavelengths_str = img_spectral.metadata['wavelength']
        try:
            if isinstance(wavelengths_str, list):
                 wavelengths = np.array([float(w) for w in wavelengths_str], dtype=np.float32)
            elif isinstance(wavelengths_str, str):
                 sep = ',' if ',' in wavelengths_str else ' '
                 wavelengths = np.array([float(w) for w in wavelengths_str.strip('{}').split(sep) if w.strip()], dtype=np.float32)
            else:
                 wavelengths = np.array(wavelengths_str, dtype=np.float32)
            source = "metadata['wavelength']"
        except (ValueError, TypeError) as e_wl:
             print(f"    Warning: Error parsing 'wavelength' from metadata: {e_wl}")
             wavelengths = None
    elif hasattr(img_spectral, 'metadata') and 'band names' in img_spectral.metadata:
         try:
             wavelengths = np.array([float(bn.split()[0]) for bn in img_spectral.metadata['band names']], dtype=np.float32)
             source = "metadata['band names'] (parsing)"
         except (ValueError, IndexError, AttributeError, TypeError):
             wavelengths = None

    if wavelengths is None or len(wavelengths) != bands:
        if wavelengths is not None: print(f"    WARNING: Number of wavelengths ({len(wavelengths)}) != number of bands ({bands}).")
        else: print(f"    WARNING: Wavelengths not found.")
        print(f"    --> Using band indices (0 to {bands-1}).")
        wavelengths = np.arange(bands, dtype=np.float32)
        source = "Band indices (fallback)"
        wavelength_units = 'index'
    else:
        wavelength_units = img_spectral.metadata.get('wavelength units', 'unknown')
        if not isinstance(wavelength_units, str) or not wavelength_units: wavelength_units = 'unknown'
        print(f"--> OK: Wavelengths ({len(wavelengths)}) found from {source}.")
    print(f"  Wavelength units: '{wavelength_units}'")

    data_description = img_spectral.metadata.get('description', f'Spectral Data Cube from {base_name}')
    if not isinstance(data_description, str): data_description = str(data_description)
    print(f"  Data description: '{data_description}'")

    original_metadata = {}
    if hasattr(img_spectral, 'metadata'):
        print("  Cleaning and copying original metadata...")
        for k, v in img_spectral.metadata.items():
            if isinstance(v, (int, float, str, bool)): original_metadata[k] = v
            elif isinstance(v, (list, tuple)):
                 try: original_metadata[k] = [float(i) if isinstance(i, np.floating) else str(i) if not isinstance(i, (int, float, str, bool)) else i for i in v]
                 except: original_metadata[k] = str(v)
            else: original_metadata[k] = str(v)
        print("--> OK: Metadata copied.")

except Exception as e:
    print(f"\n--> Unexpected error during metadata extraction: {type(e).__name__}: {e}")
    traceback.print_exc()
    exit()

# --- Create xarray DataArray (with or without Dask) ---
data_xr = None
try:
    print("\nCreating xarray DataArray...")
    if wavelengths is None: raise ValueError("'wavelengths' not defined.")
    if img_spectral is None: raise ValueError("'img_spectral' object not available.")

    # *** KEY MODIFICATION: Create wrapper around img_spectral ***
    print("  Creating wrapper for spectral object...")
    try:
        wrapped_spectral = SpectralArrayWrapper(img_spectral)
        print(f"--> OK: Wrapper created. Type: {type(wrapped_spectral)}")
    except Exception as e_wrap:
        print(f"\n--> ERROR during wrapper creation: {type(e_wrap).__name__}: {e_wrap}")
        traceback.print_exc()
        exit()

    # Use WRAPPER with Dask or directly with Xarray (if not using Dask)
    if USE_DASK:
        print(f"\n  Using Dask on wrapped object with chunks: {CHUNKS}")
        try:
            # Pass WRAPPED object to da.from_array.
            data_array_backend = da.from_array(
                wrapped_spectral,                   # Wrapped object
                chunks=CHUNKS,
                name=f"spectral-data-{base_name}"
            )
            print(f"--> OK: Dask array created (lazy) from wrapper.")
            print(f"    Dask array info: Chunks={data_array_backend.chunksize}, Shape={data_array_backend.shape}, Dtype={data_array_backend.dtype}")

        except Exception as e_dask:
            print(f"\n--> ERROR during Dask array creation from wrapper: {type(e_dask).__name__}: {e_dask}")
            traceback.print_exc()
            exit()

    else: # Don't use Dask
        print("\n  Loading data into memory (NumPy) via wrapper...")
        try:
            data_array_backend_raw = wrapped_spectral[:] # Use wrapper to read everything
            if not isinstance(data_array_backend_raw, np.ndarray):
                 print(f"    Warning: Reading from wrapper not ndarray ({type(data_array_backend_raw)}). Converting...")
                 data_array_backend = np.array(data_array_backend_raw)
            else:
                 data_array_backend = data_array_backend_raw
            if not isinstance(data_array_backend, np.ndarray):
                print(f"--> ERROR: Unable to get NumPy array. Type: {type(data_array_backend)}")
                exit()
            print(f"--> OK: Data loaded into NumPy. Shape: {data_array_backend.shape}, Type: {data_array_backend.dtype}")
        except Exception as e_load:
             print(f"\n--> ERROR during data loading into NumPy: {type(e_load).__name__}: {e_load}")
             traceback.print_exc()
             exit()

    # --- Now create xarray DataArray using Dask or NumPy backend ---
    print(f"\n  Attempting to create xarray.DataArray using backend type: {type(data_array_backend)}")
    if not hasattr(data_array_backend, 'ndim') or \
       not hasattr(data_array_backend, 'shape') or \
       not hasattr(data_array_backend, 'dtype'):
        print(f"--> INTERNAL ERROR: 'data_array_backend' object ({type(data_array_backend)}) missing required attributes.")
        exit()

    data_xr = xr.DataArray(
        data=data_array_backend, # Pass Dask or NumPy array
        coords={
            'line': np.arange(lines),
            'sample': np.arange(samples),
            'wavelength': wavelengths
        },
        dims=['line', 'sample', 'wavelength'],
        name=data_description or base_name,
        attrs=original_metadata
    )

    data_xr['line'].attrs['long_name'] = 'Spatial dimension Y (line)'
    data_xr['line'].attrs['units'] = 'pixel_index'
    data_xr['sample'].attrs['long_name'] = 'Spatial dimension X (sample)'
    data_xr['sample'].attrs['units'] = 'pixel_index'
    data_xr['wavelength'].attrs['long_name'] = 'Wavelength'
    data_xr['wavelength'].attrs['units'] = wavelength_units

    print("--> OK: xarray DataArray successfully created!")
    print("\n--- xarray DataArray Information ---")
    with xr.set_options(display_expand_data=False):
        print(data_xr)

except ValueError as ve:
     print(f"\n--> Value Error during DataArray creation: {ve}")
     exit()
except Exception as e:
    print(f"\n--> Unexpected error during DataArray creation: {type(e).__name__}: {e}")
    traceback.print_exc()
    exit()

# --- NetCDF Save ---
if data_xr is not None and NETCDF_ENGINE:
    print(f"\n--- Saving DataArray to NetCDF ({output_nc_file}) ---")
    try:
        encoding_options = {
            data_xr.name: {
                'zlib': True, 'complevel': 4,
                '_FillValue': np.nan if data_xr.dtype.kind == 'f' else None,
            }
        }
        if encoding_options[data_xr.name]['_FillValue'] is None:
             del encoding_options[data_xr.name]['_FillValue']

        print(f"  Saving with engine '{NETCDF_ENGINE}'...")
        print(f"  Encoding for '{data_xr.name}': {encoding_options.get(data_xr.name, 'Default')}")

        write_job = data_xr.to_netcdf(
            path=output_nc_file, mode='w', engine=NETCDF_ENGINE,
            encoding=encoding_options, compute=False
        )

        print("  Starting computation and writing...")
        if USE_DASK:
            with ProgressBar(): write_job.compute()
        else: write_job.compute()

        print(f"--> OK: DataArray successfully saved to '{output_nc_file}'")

    except ImportError:
         print(f"--> ERROR: NetCDF engine '{NETCDF_ENGINE}' not installed.")
    except Exception as e:
        print(f"\n--> ERROR during NetCDF save:")
        print(f"    Type: {type(e).__name__}, Msg: {e}")
        print(f"    File: {output_nc_file}. Check permissions and space.")
        traceback.print_exc()

elif data_xr is None: print("\nNetCDF save skipped: DataArray not created.")
else: print("\nNetCDF save skipped: 'netCDF4' library not installed.")

# --- Example Plotting ---
if data_xr is not None:
    try:
        print("\n--- Example: Central pixel spectrum visualization ---")
        center_line, center_sample = lines // 2, samples // 2
        print(f"  Pixel: line={center_line}, sample={center_sample}")

        pixel_spectrum_lazy = data_xr.isel(line=center_line, sample=center_sample)
        print(f"  Loading spectrum ({pixel_spectrum_lazy.size} points)...")
        pixel_spectrum = pixel_spectrum_lazy.load() # Load only this piece
        print(f"--> OK: Spectrum loaded. Data type: {type(pixel_spectrum.data)}")

        print(f"  Creating plot...")
        plt.figure(figsize=(12, 7))
        pixel_spectrum.plot(marker='.', linestyle='-', linewidth=1)

        plt.title(f"CRISM Spectrum - Pixel ({center_line}, {center_sample})\nFile: {os.path.basename(hdr_file_path)}")
        plt.xlabel(f"Wavelength ({data_xr['wavelength'].attrs.get('units', 'N/A')})")
        z_label = data_xr.name or "Value"
        data_units = data_xr.attrs.get('z plot titles', data_xr.attrs.get('data units', 'unknown units'))
        plt.ylabel(f"{z_label}\n({data_units})")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        print("--> OK: Plot ready. Showing window...")
        plt.show()

    except Exception as e:
        print(f"\n--> ERROR during spectrum visualization: {type(e).__name__}: {e}")
        traceback.print_exc()

# --- Final Notes ---
print("\n--- Note on auxiliary files (_if) ---")
print("This script loads only the main data cube.")
print("For complete analysis, consider loading auxiliary files (e.g. _if) into an xarray Dataset.")
print("\n--- Script completed ---")