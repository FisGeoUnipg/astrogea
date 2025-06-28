# -*- coding: utf-8 -*-
"""
Script to read a CRISM spectral cube (ENVI-like format),
convert it to an xarray.DataArray and save it in NetCDF format.
"""

# --- Imports ---
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Import from 'spectral' library (installed with 'pip install spectralpy')
try:
    import spectral
    from spectral import open_image
    print(f"Spectralpy library (version: {spectral.__version__}) successfully imported.")
except ImportError:
    print("\nCRITICAL ERROR: 'spectral' library not found.")
    print("This library is required to read image files.")
    print("-------------------------------------------------------------")
    print("INSTALLATION INSTRUCTIONS:")
    print("1. Make sure you have pip updated: python -m pip install --upgrade pip")
    print("2. Run in your terminal:")
    print("   python -m pip install spectralpy")
    print("-------------------------------------------------------------")
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
    print(f"\nCRITICAL ERROR: Missing dependency library: {e}")
    print("Make sure you have installed xarray, numpy, matplotlib.")
    exit()

# Import NetCDF engine (required for saving)
try:
    import netCDF4
    print(f"NetCDF engine (netCDF4) successfully imported.")
    NETCDF_ENGINE = 'netcdf4'
except ImportError:
    print("\nWARNING: 'netCDF4' library not found.")
    print("It will not be possible to save the output in NetCDF format.")
    print("To enable saving, install it with:")
    print("   python -m pip install netCDF4")
    NETCDF_ENGINE = None # Indicates we cannot save

# --- Configuration ---
hdr_file_path = 'frt000144ff_07_sr164j_mtr3.hdr'
# Define the output NetCDF filename (based on input)
base_name = os.path.splitext(os.path.basename(hdr_file_path))[0]
output_nc_file = f"{base_name}_xarray.nc"
print(f"\nPlanned NetCDF output file: '{output_nc_file}'")

# --- Verify Header File Existence ---
print(f"\nVerifying header file existence: '{hdr_file_path}'")
if not os.path.exists(hdr_file_path):
    print(f"--> ERROR: The specified header file was NOT found.")
    print(f"   Searched path: {os.path.abspath(hdr_file_path)}")
    exit()
else:
    print("--> OK: Header file found.")

# --- Load Data with Spectral ---
img_spectral = None
try:
    print(f"\nAttempting to open image file via header: {hdr_file_path}")
    img_spectral = open_image(hdr_file_path)
    print("--> OK: File successfully opened using spectral.")
    print(f"  Returned object type: {type(img_spectral)}")
    print(f"  Dimensions (lines, samples, bands): {img_spectral.shape}")
    if hasattr(img_spectral, 'metadata'):
        print(f"  Read metadata (first 10 keys): {list(img_spectral.metadata.keys())[:10]}...")
    else:
        print("  Warning: Image object does not have 'metadata' attribute.")
except FileNotFoundError:
    img_file_expected = hdr_file_path.replace('.hdr', '.img')
    if not os.path.exists(img_file_expected):
         print(f"--> ERROR: Unable to find the associated image file (.img).")
         print(f"   The spectral library expected to find: '{img_file_expected}'")
    else:
         print(f"--> ERROR: Unexplained FileNotFoundError during opening.")
    exit()
except Exception as e:
    print(f"\n--> ERROR during file opening with spectral: {type(e).__name__}: {e}")
    exit()

# --- Extract Data and Metadata ---
data_np = None
wavelengths = None
lines, samples, bands = 0, 0, 0
wavelength_units = 'unknown'
data_description = 'Spectral Data Cube'
original_metadata = {}
try:
    print("\nExtracting data and metadata from spectral object...")
    print("  Loading data into memory...")
    data_np = img_spectral.load()
    print(f"--> OK: Data loaded. NumPy shape: {data_np.shape}, Type: {data_np.dtype}")

    lines, samples, bands = img_spectral.shape
    print(f"  Dimensions: Lines={lines}, Samples={samples}, Bands={bands}")

    print("  Extracting wavelengths...")
    # ... (wavelength extraction logic) ...
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
                 wavelengths = np.array([float(w) for w in wavelengths_str.split(sep) if w.strip()], dtype=np.float32)
            else: wavelengths = np.array(wavelengths_str, dtype=np.float32)
            source = "metadata['wavelength']"
        except (ValueError, TypeError): wavelengths = None
    elif hasattr(img_spectral, 'metadata') and 'band names' in img_spectral.metadata:
         try:
             wavelengths = np.array([float(bn) for bn in img_spectral.metadata['band names']], dtype=np.float32)
             source = "metadata['band names']"
         except ValueError: wavelengths = None
    else: wavelengths = None

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
        print(f"--> OK: Wavelengths ({len(wavelengths)}) from {source}.")

    print(f"  Wavelength units: '{wavelength_units}'")

    data_description = img_spectral.metadata.get('description', 'Spectral Data Cube')
    if not isinstance(data_description, str): data_description = str(data_description)
    print(f"  Data description: '{data_description}'")

    # Preserve original metadata
    original_metadata = img_spectral.metadata if hasattr(img_spectral, 'metadata') else {}

except Exception as e:
    print(f"\n--> Unexpected error during data or metadata extraction: {type(e).__name__}: {e}")
    exit()

# --- Create xarray DataArray ---
data_xr = None
try:
    print("\nCreating xarray DataArray...")
    if data_np is None: raise ValueError("'data_np' not loaded.")
    if wavelengths is None: raise ValueError("'wavelengths' not defined.")

    data_xr = xr.DataArray(
        data=data_np,
        coords={
            'line': np.arange(lines),
            'sample': np.arange(samples),
            'wavelength': wavelengths
        },
        dims=['line', 'sample', 'wavelength'],
        name=data_description,
        attrs=original_metadata
    )

    data_xr['line'].attrs['long_name'] = 'Spatial dimension Y (line)'
    data_xr['line'].attrs['units'] = 'pixel_index'
    data_xr['sample'].attrs['long_name'] = 'Spatial dimension X (sample)'
    data_xr['sample'].attrs['units'] = 'pixel_index'
    data_xr['wavelength'].attrs['long_name'] = 'Wavelength'
    data_xr['wavelength'].attrs['units'] = wavelength_units

    print("--> OK: xarray DataArray successfully created!")
    print("\n--- Information about xarray DataArray ---")
    print(data_xr)

except ValueError as ve:
     print(f"\n--> Value Error during DataArray creation: {ve}")
     exit()
except Exception as e:
    print(f"\n--> Unexpected error during DataArray creation: {type(e).__name__}: {e}")
    exit()

# --- *** NEW SECTION: Save DataArray to NetCDF *** ---
if data_xr is not None and NETCDF_ENGINE:
    print(f"\n--- Saving DataArray to NetCDF ({output_nc_file}) ---")
    try:
        # Defining encoding is optional but useful for compression/chunking
        # Here we enable ZLIB compression level 4 for the data variable
        encoding_options = {
            data_xr.name: { # Apply to the single data variable in DataArray
                'zlib': True,      # Enable compression
                'complevel': 4,    # Compression level (1-9)
                '_FillValue': np.nan if data_xr.dtype.kind == 'f' else None # Use NaN for float, otherwise default
                # 'chunksizes': (lines // 4, samples // 4, bands) # Example of chunking (to be adapted)
            }
        }
        # If you don't want specific encoding, pass an empty dictionary: encoding={}

        print(f"  Saving in progress with engine '{NETCDF_ENGINE}'...")
        # Save the DataArray to the specified NetCDF file
        data_xr.to_netcdf(
            path=output_nc_file,
            mode='w',              # 'w' to write (overwrites if exists)
            engine=NETCDF_ENGINE,  # Engine to use ('netcdf4' or 'h5netcdf')
            encoding=encoding_options # Encoding options (compression, etc.)
        )
        print(f"--> OK: DataArray successfully saved to '{output_nc_file}'")

    except ImportError:
         # This shouldn't happen if NETCDF_ENGINE is not None, but just in case
         print(f"--> ERROR: NetCDF engine '{NETCDF_ENGINE}' is not properly installed.")
         print(f"   Install with: python -m pip install {NETCDF_ENGINE}")
    except Exception as e:
        print(f"\n--> ERROR during NetCDF saving:")
        print(f"    Error Type: {type(e).__name__}")
        print(f"    Message: {e}")
        print(f"    Output file: {output_nc_file}")
        print(f"    Check write permissions and disk space.")

elif data_xr is None:
    print("\nNetCDF saving skipped: DataArray was not created correctly.")
else: # If NETCDF_ENGINE is None
    print("\nNetCDF saving skipped: 'netCDF4' library is not installed.")

# --- Usage Example: Display a Spectrum ---
if data_xr is not None:
    try:
        print("\n--- Example: Displaying a pixel spectrum ---")
        center_line = lines // 2
        center_sample = samples // 2
        print(f"  Selected pixel index: line={center_line}, sample={center_sample}")
        pixel_spectrum = data_xr.isel(line=center_line, sample=center_sample)
        print(f"  Extracted spectrum ({pixel_spectrum.size} points). Creating plot...")

        plt.figure(figsize=(12, 7))
        pixel_spectrum.plot(marker='.', linestyle='-', linewidth=1)
        plt.title(f"CRISM Spectrum - Pixel (line={center_line}, sample={center_sample})\nSource File: {os.path.basename(hdr_file_path)}")
        plt.xlabel(f"Wavelength ({data_xr['wavelength'].attrs.get('units', 'N/A')})")
        data_units = data_xr.attrs.get('z plot titles', data_xr.attrs.get('data units', 'Value (unknown units)'))
        plt.ylabel(f"{data_xr.name}\n({data_units})")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        print("--> OK: Plot ready. Showing window...")
        plt.show()

    except Exception as e:
        print(f"\n--> ERROR during spectrum visualization: {type(e).__name__}: {e}")

# --- Final Notes ---
print("\n--- Note on auxiliary files (_if) ---")
print("\n--- Script completed ---")