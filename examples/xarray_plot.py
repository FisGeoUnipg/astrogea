# -*- coding: utf-8 -*-
"""
Script to read an xarray DataArray from a NetCDF file
and demonstrate some analysis and plotting capabilities.
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---
# Specify the path of the NetCDF file created by the previous script
# Make sure this file exists!
netcdf_file_path = 'frt000144ff_07_sr164j_mtr3_xarray_dask.nc' # Modify if needed

# --- File Existence Check ---
print(f"Checking NetCDF file existence: '{netcdf_file_path}'")
if not os.path.exists(netcdf_file_path):
    print(f"--> ERROR: NetCDF file not found.")
    print(f"   Searched path: {os.path.abspath(netcdf_file_path)}")
    print(f"   Make sure the previous script was executed successfully")
    print(f"   and the filename is correct.")
    exit()
else:
    print("--> OK: NetCDF file found.")


# --- Loading xarray Data from NetCDF ---
data_xr = None
try:
    print(f"\nLoading DataArray from '{netcdf_file_path}'...")
    # Use xr.open_dataarray because we saved a single DataArray
    # If we had saved multiple variables, we would use xr.open_dataset()
    data_xr = xr.open_dataarray(netcdf_file_path)
    print("--> OK: Data loaded successfully!")

    # Print information about the loaded DataArray
    # Note how dimensions, coordinates and attributes are preserved!
    print("\n--- Information about the loaded DataArray ---")
    print(data_xr)

except FileNotFoundError:
    # This is already handled above, but for safety
    print(f"--> ERROR: File not found (post-verification check): {netcdf_file_path}")
    exit()
except Exception as e:
    print(f"\n--> ERROR during NetCDF file loading:")
    print(f"    Error Type: {type(e).__name__}")
    print(f"    Message: {e}")
    print(f"    The file might be corrupted or not in the expected format.")
    exit()


# --- Demonstration 1: Plot spectrum of a single pixel ---
if data_xr is not None:
    print("\n--- Demonstration 1: Spectrum of a central pixel ---")
    try:
        # Get maximum dimensions from axes (more robust)
        max_line_idx = data_xr['line'].size - 1
        max_sample_idx = data_xr['sample'].size - 1
        center_line = max_line_idx // 2
        center_sample = max_sample_idx // 2

        print(f"  Selecting pixel with indices: line={center_line}, sample={center_sample}")
        # Use .isel() to select using integer indices
        pixel_spectrum = data_xr.isel(line=center_line, sample=center_sample)

        print(f"  Spectrum extracted. Creating plot...")
        plt.figure(figsize=(10, 6))
        # xarray's .plot() method automatically uses correct coordinates (wavelength)
        pixel_spectrum.plot(marker='.', linestyle='-')

        plt.title(f"Pixel Spectrum (line={center_line}, sample={center_sample})")
        plt.xlabel(f"Wavelength ({data_xr['wavelength'].attrs.get('units', 'N/A')})")
        plt.ylabel(f"{data_xr.name} ({data_xr.attrs.get('data units', 'Data Value')})")
        plt.grid(True, linestyle=':')
        plt.tight_layout()
        # plt.show() # Show immediately or wait until the end

    except IndexError:
         print(f"--> ERROR: Indices ({center_line}, {center_sample}) out of bounds.")
    except Exception as e:
        print(f"--> ERROR during demonstration 1: {type(e).__name__}: {e}")


# --- Demonstration 2: Plot spatial image (slice at one wavelength) ---
if data_xr is not None:
    print("\n--- Demonstration 2: Image at a specific wavelength ---")
    try:
        # Choose a target wavelength (e.g., near 1000 nm or 1 micron)
        # Assume units are nm; modify if they are microns or other
        target_wavelength = 1000.0
        units = data_xr['wavelength'].attrs.get('units', 'unknown')
        # You might want to convert target_wavelength if units are different (e.g., microns)
        # if units.lower() == 'microns' or units.lower() == 'um':
        #    target_wavelength = 1.0

        print(f"  Selecting spatial slice closest to {target_wavelength} {units}")
        # Use .sel() to select based on 'wavelength' coordinate value
        # 'method="nearest"' finds the band closest to the target value
        spatial_slice = data_xr.sel(wavelength=target_wavelength, method="nearest")

        # Get the actually selected wavelength
        actual_wavelength = spatial_slice['wavelength'].item() # .item() to extract scalar value

        print(f"  Actually selected wavelength: {actual_wavelength:.2f} {units}")
        print(f"  Data extracted (shape: {spatial_slice.shape}). Creating spatial map...")

        plt.figure(figsize=(8, 8))
        # xarray's .plot() for 2D data uses imshow and adds a colorbar
        spatial_slice.plot(cmap='viridis') # You can choose other colormaps: 'gray', 'jet', etc.

        plt.title(f"Spatial Map at ~{actual_wavelength:.0f} {units}")
        # xarray usually labels axes automatically, but you can customize:
        # plt.xlabel(f"Sample Index ({data_xr['sample'].attrs.get('units', 'N/A')})")
        # plt.ylabel(f"Line Index ({data_xr['line'].attrs.get('units', 'N/A')})")
        plt.tight_layout()
        # plt.show() # Show immediately or wait until the end

    except KeyError:
         print(f"--> ERROR: 'wavelength' coordinate not found in DataArray.")
    except Exception as e:
        print(f"--> ERROR during demonstration 2: {type(e).__name__}: {e}")


# --- Demonstration 3: Calculate and plot mean spectrum ---
if data_xr is not None:
    print("\n--- Demonstration 3: Mean spectrum of the entire image ---")
    try:
        print(f"  Calculating mean along 'line' and 'sample' dimensions...")
        # Use .mean() specifying dimensions to aggregate (collapse)
        # skipna=True ignores NaN values (if present) in mean calculation
        mean_spectrum = data_xr.mean(dim=['line', 'sample'], skipna=True)

        print(f"  Mean spectrum calculated (shape: {mean_spectrum.shape}). Creating plot...")
        plt.figure(figsize=(10, 6))
        # Plot the resulting mean spectrum
        mean_spectrum.plot(color='red')

        plt.title("Mean Spectrum of the Image")
        plt.xlabel(f"Wavelength ({data_xr['wavelength'].attrs.get('units', 'N/A')})")
        plt.ylabel(f"Mean of {data_xr.name} ({data_xr.attrs.get('data units', 'Data Value')})")
        plt.grid(True, linestyle=':')
        plt.tight_layout()
        # plt.show() # Show immediately or wait until the end

    except Exception as e:
        print(f"--> ERROR during demonstration 3: {type(e).__name__}: {e}")


# --- Show all plots ---
if data_xr is not None:
    print("\nDisplaying plots...")
    plt.show()
else:
    print("\nNo plots to display due to previous errors.")

print("\n--- Analysis script completed ---")