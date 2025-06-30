# -*- coding: utf-8 -*-
"""
Script to read a CRISM spectral cube (ENVI-like format)
and convert it to an xarray.DataArray using the spectralpy library.
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
    print("If the error persists, check your Python environment.")
    exit() # Exit the script if the fundamental library is missing
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
    print("Make sure you have installed xarray, numpy, matplotlib:")
    print("   python -m pip install xarray numpy matplotlib")
    exit()


# --- Configuration ---
# Make sure this file is in the same directory as the script
# or provide the complete path.
# This is the header for *scientific* data.
hdr_file_path = 'frt000144ff_07_sr164j_mtr3.hdr'

# --- Check Header File Existence ---
print(f"\nChecking header file existence: '{hdr_file_path}'")
if not os.path.exists(hdr_file_path):
    print(f"--> ERROR: The specified header file was NOT found.")
    print(f"   Path searched: {os.path.abspath(hdr_file_path)}")
    print("   Make sure the filename is correct and it's in")
    print("   the right directory or provide the absolute path.")
    exit()
else:
    print("--> OK: Header file found.")

# --- Load Data with Spectral ---
img_spectral = None # Initialize to None
try:
    print(f"\nAttempting to open image file via header: {hdr_file_path}")
    # Use the 'open_image' function imported from 'spectral'
    # This function reads the header and opens the associated .img file
    img_spectral = open_image(hdr_file_path)

    print("--> OK: File successfully opened using spectral.")
    print(f"  Returned object type: {type(img_spectral)}")
    print(f"  Dimensions (lines, samples, bands): {img_spectral.shape}")

    # Check and print some metadata if available
    if hasattr(img_spectral, 'metadata'):
        print(f"  Metadata read (first 10 keys): {list(img_spectral.metadata.keys())[:10]}...")
    else:
        print("  Warning: Image object does not have 'metadata' attribute.")

except FileNotFoundError:
    # This error can occur if the .img file is not found
    img_file_expected = hdr_file_path.replace('.hdr', '.img')
    if not os.path.exists(img_file_expected):
         print(f"--> ERROR: Unable to find the associated image file (.img).")
         print(f"   The spectral library expected to find: '{img_file_expected}'")
         print(f"   but it doesn't exist in the same directory as the header.")
         print(f"   Header path: {os.path.abspath(hdr_file_path)}")
    else:
         print(f"--> ERROR: Unexplained FileNotFoundError during opening.")
         print(f"   The header file '{hdr_file_path}' and image file '{img_file_expected}' seem to exist.")
         print(f"   Check file permissions or any header issues.")
    exit()
except Exception as e:
    # Catch other errors during opening (e.g., malformed header, I/O errors)
    print(f"\n--> ERROR during file opening with spectral:")
    print(f"    Error Type: {type(e).__name__}")
    print(f"    Message: {e}")
    print(f"    File involved: {hdr_file_path}")
    print("    Check that the header file is not corrupted and the .img file is accessible.")
    exit()


# --- Extract Data and Metadata ---
data_np = None
wavelengths = None
try:
    print("\nExtracting data and metadata from spectral object...")
    # Load the entire data cube into a NumPy array
    # WARNING: Memory intensive! For huge files, consider Dask.
    print("  Loading data into memory (might take time)...")
    data_np = img_spectral.load()
    print(f"--> OK: Data loaded. NumPy array shape: {data_np.shape}")
    print(f"    NumPy data type: {data_np.dtype}")

    # Extract dimensions
    lines, samples, bands = img_spectral.shape
    print(f"  Dimensions: Lines={lines}, Samples={samples}, Bands={bands}")

    # Extract wavelengths (with various fallbacks)
    print("  Extracting wavelengths...")
    if hasattr(img_spectral, 'bands') and hasattr(img_spectral.bands, 'centers') and img_spectral.bands.centers:
        wavelengths = np.array(img_spectral.bands.centers, dtype=np.float32)
        source = "'img_spectral.bands.centers'"
    elif hasattr(img_spectral, 'metadata') and 'wavelength' in img_spectral.metadata:
        wavelengths_str = img_spectral.metadata['wavelength']
        try:
            if isinstance(wavelengths_str, list):
                 wavelengths = np.array([float(w) for w in wavelengths_str], dtype=np.float32)
            elif isinstance(wavelengths_str, str):
                 # Try splitting by comma or space
                 sep = ',' if ',' in wavelengths_str else ' '
                 wavelengths = np.array([float(w) for w in wavelengths_str.split(sep) if w.strip()], dtype=np.float32)
            else: # Try direct conversion if it's a strange type
                 wavelengths = np.array(wavelengths_str, dtype=np.float32)
            source = "metadata['wavelength']"
        except (ValueError, TypeError) as parse_err:
             print(f"    WARNING: Error parsing 'wavelength' from metadata: {parse_err}. Fallback...")
             wavelengths = None
             source = "Error parsing metadata['wavelength']"
    elif hasattr(img_spectral, 'metadata') and 'band names' in img_spectral.metadata:
         try:
             wavelengths = np.array([float(bn) for bn in img_spectral.metadata['band names']], dtype=np.float32)
             source = "metadata['band names']"
         except ValueError:
             print("    WARNING: 'band names' in metadata don't seem numeric. Fallback...")
             wavelengths = None
             source = "Error parsing metadata['band names']"
    else:
        wavelengths = None
        source = "No source found"

    # Final fallback if not found or if number doesn't match
    if wavelengths is None or len(wavelengths) != bands:
        if wavelengths is not None: # If found but number is wrong
             print(f"    WARNING: Number of wavelengths found ({len(wavelengths)}) doesn't match number of bands ({bands}).")
        else: # If not found at all
             print(f"    WARNING: Wavelength information not found in standard metadata.")
        print(f"    --> Using band indices (0 to {bands-1}) as 'wavelength' coordinates.")
        wavelengths = np.arange(bands, dtype=np.float32)
        source = "Band indices (fallback)"
        wavelength_units = 'index'
    else:
        # Extract units only if wavelengths are valid
        wavelength_units = img_spectral.metadata.get('wavelength units', 'unknown')
        if not isinstance(wavelength_units, str) or not wavelength_units:
             wavelength_units = 'unknown' # Sanitize invalid/empty units
        print(f"--> OK: Wavelengths extracted from {source}. Number: {len(wavelengths)}")

    print(f"  Wavelength units: '{wavelength_units}'")

    # Extract data description
    data_description = img_spectral.metadata.get('description', 'Spectral Data Cube')
    if not isinstance(data_description, str): data_description = str(data_description) # Ensure it's a string
    print(f"  Data description: '{data_description}'")

except AttributeError as e:
    print(f"\n--> ERROR during access to data/metadata from spectral object:")
    print(f"    Message: {e}")
    print(f"    The 'img_spectral' object (type: {type(img_spectral)}) might not have been loaded correctly.")
    exit()
except Exception as e:
    print(f"\n--> Unexpected error during data or metadata extraction:")
    print(f"    Error Type: {type(e).__name__}")
    print(f"    Message: {e}")
    exit()


# --- Create xarray DataArray ---
data_xr = None
try:
    print("\nCreating xarray DataArray...")
    # Verify that data has been loaded
    if data_np is None:
        raise ValueError("The NumPy array 'data_np' was not loaded correctly.")
    if wavelengths is None:
        raise ValueError("The 'wavelengths' coordinates were not properly defined.")

    # Retrieve all original metadata for attributes
    original_metadata = img_spectral.metadata if hasattr(img_spectral, 'metadata') else {}

    data_xr = xr.DataArray(
        data=data_np,                   # The NumPy array with data
        coords={                        # Dictionary of coordinates
            'line': np.arange(lines),       # Coordinate for lines (Y dimension)
            'sample': np.arange(samples),   # Coordinate for samples (X dimension)
            'wavelength': wavelengths       # Coordinate for wavelengths
        },
        dims=['line', 'sample', 'wavelength'], # Names of dimensions in data order
        name=data_description,          # Name of data variable (from metadata 'description')
        attrs=original_metadata         # Copy all original metadata from ENVI header
    )

    # Add specific and descriptive attributes to coordinates
    data_xr['line'].attrs['long_name'] = 'Spatial dimension Y (line)'
    data_xr['line'].attrs['units'] = 'pixel_index'
    data_xr['sample'].attrs['long_name'] = 'Spatial dimension X (sample)'
    data_xr['sample'].attrs['units'] = 'pixel_index'
    data_xr['wavelength'].attrs['long_name'] = 'Wavelength'
    data_xr['wavelength'].attrs['units'] = wavelength_units

    print("--> OK: xarray DataArray successfully created!")
    print("\n--- Information about xarray DataArray ---")
    print(data_xr) # Print a representation of the xarray object

except ValueError as ve:
     print(f"\n--> Value Error during xarray DataArray creation:")
     print(f"    Message: {ve}")
     exit()
except Exception as e:
    print(f"\n--> Unexpected error during xarray DataArray creation:")
    print(f"    Error Type: {type(e).__name__}")
    print(f"    Message: {e}")
    exit()


# --- Usage Example: Display a Spectrum ---
if data_xr is not None:
    try:
        print("\n--- Example: Displaying a pixel spectrum ---")
        # Select a pixel (e.g., approximate center of the image)
        # We use .isel to select via integer index (safer here)
        center_line = lines // 2
        center_sample = samples // 2
        print(f"  Selecting pixel at index coordinates: line={center_line}, sample={center_sample}")

        # Extract the spectrum for that pixel
        pixel_spectrum = data_xr.isel(line=center_line, sample=center_sample)
        print(f"  Spectrum extracted. Number of points: {pixel_spectrum.size}")

        # Create the plot using xarray's integrated plotting
        print("  Creating plot...")
        plt.figure(figsize=(12, 7)) # Plot dimensions

        # The .plot() method of xarray automatically uses the correct coordinates
        pixel_spectrum.plot(marker='.', linestyle='-', linewidth=1)

        plt.title(f"CRISM Spectrum - Pixel (line={center_line}, sample={center_sample})\nFile: {os.path.basename(hdr_file_path)}")
        plt.xlabel(f"Wavelength ({data_xr['wavelength'].attrs.get('units', 'N/A')})") # Gets units from attributes
        # Look for data units in metadata (often not explicit in ENVI)
        data_units = data_xr.attrs.get('z plot titles', # Sometimes here in ENVI
                                       data_xr.attrs.get('data units', # Or here
                                                        'Value (unknown units)'))
        plt.ylabel(f"{data_xr.name}\n({data_units})")
        plt.grid(True, linestyle='--', alpha=0.6) # Background grid
        plt.tight_layout() # Adjust spacing
        print("--> OK: Plot ready. Showing window...")
        plt.show() # Show the plot window

    except IndexError:
         print(f"--> Index Error during pixel selection ({center_line}, {center_sample}).")
         print(f"    Image dimensions are ({lines}, {samples}). Check indices.")
    except Exception as e:
        print(f"\n--> Error during spectrum visualization:")
        print(f"    Error Type: {type(e).__name__}")
        print(f"    Message: {e}")

# --- Notes on _if_ (auxiliary) files ---
print("\n--- Note on auxiliary (_if) files ---")
print("The script has only processed _sr files (scientific data).")
print("The 'frt000144ff_07_if164j_mtr3.*' files contain auxiliary data (e.g., geometry).")
print("They could be read separately (if in ENVI format) and integrated")
print("into the xarray DataArray as additional coordinates or in an xarray.Dataset.")

print("\n--- Script completed ---")

# --- ESEMPIO: Continuum Removal + salvataggio NetCDF con WCS (NumPy e Dask) ---
if __name__ == "__main__":
    from astrogea.core import continuum_removal, continuum_to_xarray_wcs
    from astrogea.wcs_utils import parse_envi_map_info_list
    import numpy as np
    import xarray as xr
    # Dati sintetici
    img = np.random.rand(5, 5, 20) * 1000
    wavelength = np.linspace(1000, 2500, 20)
    MIN, MAX = 1200, 2200
    # --- Versione NumPy ---
    result, x = continuum_removal(img, wavelength, MIN, MAX)
    map_info = [
        "UTM", 1.0, 2.0, 100.0, 200.0, 30.0, 30.0, "meters"
    ]
    parsed_map_info = parse_envi_map_info_list(map_info)
    da = continuum_to_xarray_wcs(result, x, parsed_map_info)
    out_path = "out_continuum.nc"
    da.to_netcdf(out_path)
    print(f"File NetCDF con WCS salvato (NumPy): {out_path}")
    loaded = xr.open_dataarray(out_path)
    print("Attributi del file NumPy:", loaded.attrs)
    # --- Versione Dask ---
    try:
        import dask.array as da_dask
        dask_img = da_dask.from_array(img, chunks=(2, 2, 20))
        result_dask, x_dask = continuum_removal(dask_img, wavelength, MIN, MAX, use_dask=True)
        computed = result_dask.compute()
        da_dask_xr = continuum_to_xarray_wcs(computed, x_dask, parsed_map_info)
        out_path_dask = "out_continuum_dask.nc"
        da_dask_xr.to_netcdf(out_path_dask)
        print(f"File NetCDF con WCS salvato (Dask): {out_path_dask}")
        loaded_dask = xr.open_dataarray(out_path_dask)
        print("Attributi del file Dask:", loaded_dask.attrs)
    except ImportError:
        print("Dask non disponibile, esempio solo NumPy.")