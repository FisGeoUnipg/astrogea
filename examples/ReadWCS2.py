# -*- coding: utf-8 -*-
"""
Script to read a combined CRISM NetCDF Dataset and create
various types of demonstration plots: pixel spectrum, spatial profile,
approximate RGB image, histogram.
"""

# --- Imports ---
import os
import time
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import warnings
import traceback
import ast # Per valutare il dizionario WCS dalla stringa (se usato)

# Importa librerie WCS (opzionale, per selezionare pixel/regioni tramite coords)
try:
    from astropy.wcs import WCS
    from astropy import units as u
    ASTROPY_AVAILABLE = True
except ImportError:
    print("ATTENZIONE: Libreria 'astropy' non trovata. Funzionalità WCS non disponibili.")
    ASTROPY_AVAILABLE = False

# --- Helper Function for RGB Normalization ---
def scale_data_rgb(data_array):
    """Normalize an array (e.g. R, G or B channel) for 0-1 display."""
    # Use percentile for robustness to outliers
    min_val, max_val = np.nanpercentile(data_array, [1, 99])
    if max_val <= min_val: # Avoid division by zero or constant values
        max_val = np.nanmax(data_array)
        min_val = np.nanmin(data_array)
        if max_val <= min_val: return np.zeros_like(data_array) # Constant or empty image
    
    scaled = (data_array - min_val) / (max_val - min_val)
    return np.clip(scaled, 0, 1) # Ensure it's between 0 and 1

# --- Configuration ---
# !!! MODIFY HERE to point to your NetCDF file !!!
netcdf_file_path = 'dati/FRT00006fbd/frt00006fbd_07_sr164j_mtr3_dataset.nc'

# Parameters for plots
pixel_line_for_spectrum = 250
pixel_sample_for_spectrum = 300

line_for_profile = 250  # Image line for spatial profile
geometry_band_for_profile = 0 # IF band index for profile and histogram (0=Incidence?)

# Spectral band indices for RGB (APPROXIMATE - adjust based on your data!)
# Find indices close to: Red (~0.7 µm), Green (~0.55 µm), Blue (~0.45 µm)
# These are just generic examples, you need to find the correct indices for your wavelengths!
approx_red_wl = 0.70
approx_green_wl = 0.55
approx_blue_wl = 0.45

# --- Start Process ---
script_start_time = time.perf_counter()
xr_dataset = None
wcs_object = None

try:
    print(f"\n1. Reading NetCDF Dataset: '{netcdf_file_path}'")
    if not os.path.exists(netcdf_file_path): raise FileNotFoundError(f"File not found: {netcdf_file_path}")
    xr_dataset = xr.open_dataset(netcdf_file_path, chunks='auto')
    print("--> OK: Dataset loaded."); print(xr_dataset)

    # 2. Optional: WCS Reconstruction (for references or future selection)
    print("\n2. Checking/Reconstructing WCS...")
    has_wcs = xr_dataset.attrs.get('has_wcs', 0) == 1
    if has_wcs and ASTROPY_AVAILABLE:
        wcs_header_str = xr_dataset.attrs.get('wcs_header_dict')
        if wcs_header_str and isinstance(wcs_header_str, str):
            try:
                wcs_header_dict = ast.literal_eval(wcs_header_str)
                wcs_object = WCS(wcs_header_dict)
                print("--> OK: WCS object reconstructed.")
            except Exception as e_wcs: print(f"   WARN: WCS reconstruction failed: {e_wcs}"); wcs_object = None
        else: print("   WARN: wcs_header_dict attribute not found or not a string."); wcs_object = None
    elif has_wcs and not ASTROPY_AVAILABLE: print("   WARN: WCS info present but Astropy not available.")
    else: print("   INFO: No WCS information found in file.")

    # --- 3. Generate Plots ---
    print("\n3. Generating Plots...")

    # --- Plot 1: Single Pixel Spectrum ---
    try:
        print(f"\n   Plotting Spectrum: Pixel (line={pixel_line_for_spectrum}, sample={pixel_sample_for_spectrum})...")
        # Select pixel and load spectral data
        pixel_spectrum = xr_dataset['spectral_data'].sel(
            line=pixel_line_for_spectrum,
            sample=pixel_sample_for_spectrum
        ).load() # Load data for this pixel

        plt.figure(figsize=(10, 6))
        pixel_spectrum.plot(marker='.', linestyle='-', linewidth=1) # Use xarray's built-in plot
        plt.title(f"Pixel Spectrum (L={pixel_line_for_spectrum}, S={pixel_sample_for_spectrum})")
        plt.xlabel(f"Wavelength ({pixel_spectrum['wavelength'].attrs.get('units', '?')})")
        plt.ylabel(f"{pixel_spectrum.name}\n({pixel_spectrum.attrs.get('description', 'Value')})")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        print("--> OK: Spectrum plot generated.")
    except Exception as e: print(f"   ERROR Spectrum Plot: {type(e).__name__}: {e}"); traceback.print_exc()

    # --- Plot 2: Geometry Spatial Profile ---
    try:
        print(f"\n   Plotting Geometry Profile: Line={line_for_profile}, IF Band={geometry_band_for_profile}...")
        geom_var = xr_dataset['geometry_angles']
        if not (0 <= geometry_band_for_profile < geom_var.sizes['if_band']):
            print(f"   WARN: IF band index {geometry_band_for_profile} not valid. Using 0.")
            geometry_band_for_profile = 0

        # Select geometry band and line, load data
        profile_data = geom_var.isel(
            if_band=geometry_band_for_profile,
            line=line_for_profile
        ).load()
        if_band_name = str(profile_data['if_band'].item())

        plt.figure(figsize=(10, 5))
        profile_data.plot(linewidth=1.5)
        plt.title(f"'{if_band_name}' Profile along Line {line_for_profile}")
        plt.xlabel("Sample Index")
        plt.ylabel(f"Angle Value ({if_band_name})")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        print("--> OK: Profile plot generated.")
    except Exception as e: print(f"   ERROR Profile Plot: {type(e).__name__}: {e}"); traceback.print_exc()

    # --- Plot 3: Approximate RGB Image ---
    try:
        print(f"\n   Plotting Approximate RGB Image...")
        spec_var = xr_dataset['spectral_data']
        wavelengths = spec_var['wavelength'].values

        # Find indices of bands closest to target R, G, B
        # This assumes wavelengths are ordered
        idx_r = np.argmin(np.abs(wavelengths - approx_red_wl))
        idx_g = np.argmin(np.abs(wavelengths - approx_green_wl))
        idx_b = np.argmin(np.abs(wavelengths - approx_blue_wl))

        print(f"   Selected RGB indices: R={idx_r} (~{wavelengths[idx_r]:.3f}), G={idx_g} (~{wavelengths[idx_g]:.3f}), B={idx_b} (~{wavelengths[idx_b]:.3f})")
        if len(set([idx_r, idx_g, idx_b])) < 3: print("   WARN: RGB indices not unique! Image might look strange.")

        # Select and load the 3 bands (might require lots of memory!)
        print("   Loading RGB bands (might take time/memory)...")
        rgb_bands_data = spec_var.isel(wavelength=[idx_r, idx_g, idx_b]).load()

        # Normalize each channel to 0-1 for display
        print("   Normalizing RGB channels...")
        r_scaled = scale_data_rgb(rgb_bands_data.sel(wavelength=wavelengths[idx_r]).values)
        g_scaled = scale_data_rgb(rgb_bands_data.sel(wavelength=wavelengths[idx_g]).values)
        b_scaled = scale_data_rgb(rgb_bands_data.sel(wavelength=wavelengths[idx_b]).values)

        # Combine channels into RGB array (lines, samples, 3)
        rgb_image = np.stack([r_scaled, g_scaled, b_scaled], axis=-1)

        print("   Creating RGB plot...")
        plt.figure(figsize=(9, 8))
        plt.imshow(rgb_image, origin='lower', aspect='equal') # origin='lower' is standard for geospatial images
        plt.title(f"Approximate RGB Image (R={wavelengths[idx_r]:.2f}, G={wavelengths[idx_g]:.2f}, B={wavelengths[idx_b]:.2f})")
        plt.xlabel("Sample Index")
        plt.ylabel("Line Index")
        # Add a note about normalization
        plt.text(0.01, 0.01, 'Scaling: 1-99 percentile', transform=plt.gca().transAxes, color='white', backgroundcolor='black', fontsize=8)
        plt.tight_layout()
        print("--> OK: RGB plot generated.")
    except Exception as e: print(f"   ERROR RGB Plot: {type(e).__name__}: {e}"); traceback.print_exc()

    # --- Plot 4: Geometry Angle Histogram ---
    try:
        print(f"\n   Plotting Geometry Histogram: IF Band={geometry_band_for_profile}...")
        geom_var = xr_dataset['geometry_angles']
        if not (0 <= geometry_band_for_profile < geom_var.sizes['if_band']): geometry_band_for_profile = 0

        # Select geometry band and load ALL data (might require memory)
        # If using Dask and dataset is large, you might want to use dask.array.histogram
        print("   Loading geometry band data for histogram...")
        geom_data_1band = geom_var.isel(if_band=geometry_band_for_profile).load()
        if_band_name = str(geom_data_1band['if_band'].item())

        # Extract NumPy values, flatten and remove NaN/Inf
        values = geom_data_1band.values.ravel()
        valid_values = values[np.isfinite(values)]

        if len(valid_values) > 0:
            print(f"   Creating histogram for '{if_band_name}'...")
            plt.figure(figsize=(8, 5))
            plt.hist(valid_values, bins=100, color='skyblue', edgecolor='black') # Increased number of bins
            plt.title(f"Value Distribution: {if_band_name}")
            plt.xlabel(f"Angle Value ({if_band_name})")
            plt.ylabel("Number of Pixels")
            plt.grid(True, axis='y', linestyle='--', alpha=0.6)
            mean_val = np.mean(valid_values)
            std_val = np.std(valid_values)
            plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_val:.2f}')
            plt.legend()
            plt.tight_layout()
            print("--> OK: Histogram plot generated.")
        else:
            print(f"   WARN: No valid values found for band '{if_band_name}' for histogram.")

    except Exception as e: print(f"   ERROR Histogram Plot: {type(e).__name__}: {e}"); traceback.print_exc()


    # --- Show all plots ---
    print("\n4. Showing generated plot windows...")
    plt.show()


except FileNotFoundError as e: print(f"\nERROR: File not found - {e}")
except ImportError as e: print(f"\nERROR: Missing library - {e}")
except Exception as e:
    print(f"\n--> UNEXPECTED ERROR in read/plot process:")
    print(f"    Type: {type(e).__name__}, Msg: {e}")
    traceback.print_exc()
finally:
    if xr_dataset is not None:
        try: xr_dataset.close()
        except: pass
    script_duration = time.perf_counter() - script_start_time
    print(f"\n--- Total Read/Plot Execution Time: {script_duration:.2f} seconds ---")
    print("\n--- Read/Plot Script Completed ---")