# -*- coding: utf-8 -*-
"""
Script to read a combined CRISM NetCDF Dataset (with optional WCS),
reconstruct the Astropy WCS object from metadata and demonstrate its
usage to obtain coordinates or plot georeferenced data.
"""

# --- Imports ---
import os
import time
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import warnings
import traceback
import ast # Per valutare il dizionario WCS dalla stringa

# Importa librerie WCS e plotting
try:
    from astropy.wcs import WCS
    from astropy.visualization.wcsaxes import WCSAxes
    from astropy import units as u
    print("Astropy libraries (WCS, visualization, units) imported.")
except ImportError:
    print("ERROR: 'astropy' library not found or incomplete. Install with: pip install astropy"); exit()

# --- Configuration ---
# !!! MODIFY HERE to point to the NetCDF file created by Program 1 !!!
netcdf_file_path = 'frt000144ff_07_sr164j_mtr3_dataset.nc'

# Spectral band/wavelength indices to plot (examples)
spectral_band_index_to_plot = 150  # SR band index
geometry_band_index_to_plot = 0    # IF band index (0=Incidence?, 1=Emission?, 2=Phase?)

# Example pixel/coordinates to convert
example_pixel_line = 250
example_pixel_sample = 300
# Coordinate di esempio (MODIFICA se necessario, usa metri se WCS è in metri)
example_coord_x = -7649000.0 # Esempio Easting (m) o Lon (deg)
example_coord_y = 1110000.0  # Esempio Northing (m) o Lat (deg)

# --- Start Process ---
script_start_time = time.perf_counter()
xr_dataset = None
wcs_object = None

try:
    print(f"\n1. Reading NetCDF Dataset: '{netcdf_file_path}'")
    if not os.path.exists(netcdf_file_path):
        raise FileNotFoundError(f"NetCDF file not found: {netcdf_file_path}")

    # Open the Dataset (use chunks='auto' for potential Dask)
    xr_dataset = xr.open_dataset(netcdf_file_path, chunks='auto')
    print("--> OK: NetCDF file loaded as xarray.Dataset.")
    print("\n--- Read Dataset Info ---")
    print(xr_dataset)

    # 2. WCS Object Reconstruction
    print("\n2. Attempting WCS reconstruction from global metadata...")
    # Check the INTEGER has_wcs flag
    if xr_dataset.attrs.get('has_wcs', 0) == 1:
        wcs_header_str = xr_dataset.attrs.get('wcs_header_dict')
        if wcs_header_str and isinstance(wcs_header_str, str):
            print("   Found 'wcs_header_dict' attribute (string).")
            try:
                print("   Evaluating WCS dictionary from string...")
                # Use ast.literal_eval to convert string to dictionary
                # Safer than eval()
                wcs_header_dict = ast.literal_eval(wcs_header_str)

                if isinstance(wcs_header_dict, dict):
                    # Recreate WCS object from dictionary
                    print("   Creating WCS object from dictionary...")
                    wcs_object = WCS(wcs_header_dict)
                    print("--> OK: Astropy WCS object successfully reconstructed.")
                    print(wcs_object) # Print WCS summary
                else:
                    print("   ERROR: String evaluation did not produce a dictionary.")
                    wcs_object = None

            except SyntaxError as e_eval:
                 print(f"   ERROR: 'wcs_header_dict' string not valid for ast.literal_eval: {e_eval}")
                 wcs_object = None
            except Exception as e_wcs:
                print(f"   ERROR during WCS reconstruction: {type(e_wcs).__name__}: {e_wcs}")
                wcs_object = None
        else:
            print("   'wcs_header_dict' attribute not found or not a string.")
            wcs_object = None
    else:
        print("   'has_wcs' attribute not found or is 0. No WCS information to reconstruct.")
        wcs_object = None

    # 3. WCS Usage Demonstration (if available)
    if wcs_object:
        print("\n3. Using WCS:")
        wcs_units = wcs_object.wcs.cunit # Get units ('m' or 'deg')

        # Example: Pixel -> World Coordinates
        try:
            print(f"\n   Pixel -> World Conversion:")
            # pixel_to_world expects (x, y) i.e. (sample, line) 0-based
            world_coords = wcs_object.pixel_to_world(example_pixel_sample, example_pixel_line)
            coord1, coord2 = world_coords[0], world_coords[1]
            print(f"   Pixel (line={example_pixel_line}, sample={example_pixel_sample}) corresponds to:")
            print(f"   Coord1 ({wcs_object.wcs.ctype[0]}): {coord1:.4f} {wcs_units[0]}")
            print(f"   Coord2 ({wcs_object.wcs.ctype[1]}): {coord2:.4f} {wcs_units[1]}")
        except Exception as e: print(f"   ERROR pixel_to_world conversion: {e}")

        # Example: World -> Pixel Coordinates
        try:
            print(f"\n   World -> Pixel Conversion:")
            # world_to_pixel expects (Coord1, Coord2) with Astropy units
            # Create units dynamically
            unit1 = u.Unit(wcs_units[0])
            unit2 = u.Unit(wcs_units[1])
            pixel_coords = wcs_object.world_to_pixel(example_coord_x * unit1, example_coord_y * unit2)
            sample_idx, line_idx = int(round(pixel_coords[0])), int(round(pixel_coords[1])) # 0-based indices
            print(f"   Coordinates (X={example_coord_x} {unit1}, Y={example_coord_y} {unit2}) approximately correspond to:")
            print(f"   Pixel (line={line_idx}, sample={sample_idx}) (0-based indices)")
            # Verify index validity
            if 0 <= line_idx < xr_dataset.dims['line'] and 0 <= sample_idx < xr_dataset.dims['sample']: print("   (Valid indices)")
            else: print("   (WARNING: Indices out of bounds!)")
        except Exception as e: print(f"   ERROR world_to_pixel conversion: {e}")

        # --- Example: Georeferenced Plotting ---
        print("\n   Generating Georeferenced Plots...")

        # Plot Spectral Band (_sr_)
        try:
            print(f"   Plotting SR Data (band index {spectral_band_index_to_plot})...")
            sr_var = xr_dataset['spectral_data']
            if not (0 <= spectral_band_index_to_plot < sr_var.sizes['wavelength']):
                 print(f"   SR band index {spectral_band_index_to_plot} not valid. Using index 0."); spectral_band_index_to_plot = 0

            print(f"   Loading SR band data...")
            data_slice_sr = sr_var.isel(wavelength=spectral_band_index_to_plot).load() # Load data
            wavelength_val = data_slice_sr['wavelength'].item()
            wl_units = data_slice_sr['wavelength'].attrs.get('units', '?')

            print("   Creating SR figure and WCS axes...")
            fig_sr = plt.figure(figsize=(10, 9))
            ax_sr = WCSAxes(fig_sr, [0.1, 0.1, 0.8, 0.8], wcs=wcs_object)
            fig_sr.add_axes(ax_sr)

            print("   Plotting SR image...")
            im_sr = ax_sr.imshow(data_slice_sr.data, cmap='viridis', origin='lower', aspect='equal') # aspect='equal' if pixels are square
            ax_sr.grid(color='white', ls='dotted', alpha=0.7)
            ax_sr.set_xlabel(f"{wcs_object.wcs.ctype[0]} [{wcs_units[0]}]")
            ax_sr.set_ylabel(f"{wcs_object.wcs.ctype[1]} [{wcs_units[1]}]")
            cbar_sr = fig_sr.colorbar(im_sr, ax=ax_sr, fraction=0.046, pad=0.04)
            cbar_sr.set_label(f"{sr_var.name} @ {wavelength_val:.3f} {wl_units}")
            plt.suptitle(f"Spectral Data (WCS) - Band {spectral_band_index_to_plot}\n{os.path.basename(netcdf_file_path)}", y=0.95)
            print("--> OK: SR plot generated.")

        except Exception as e_plot_sr: print(f"   ERROR SR plot: {type(e_plot_sr).__name__}: {e_plot_sr}"); traceback.print_exc()

        # Plot Geometry Band (_if_)
        try:
            print(f"\n   Plotting Geometry Data (band index {geometry_band_index_to_plot})...")
            if_var = xr_dataset['geometry_angles']
            if not (0 <= geometry_band_index_to_plot < if_var.sizes['if_band']):
                 print(f"   IF band index {geometry_band_index_to_plot} not valid. Using index 0."); geometry_band_index_to_plot = 0

            print(f"   Loading IF band data...")
            data_slice_if = if_var.isel(if_band=geometry_band_index_to_plot).load() # Load data
            if_band_name = str(data_slice_if['if_band'].item()) # Get band name

            print("   Creating IF figure and WCS axes...")
            fig_if = plt.figure(figsize=(10, 9))
            ax_if = WCSAxes(fig_if, [0.1, 0.1, 0.8, 0.8], wcs=wcs_object)
            fig_if.add_axes(ax_if)

            print("   Plotting IF image...")
            im_if = ax_if.imshow(data_slice_if.data, cmap='magma', origin='lower', aspect='equal')
            ax_if.grid(color='white', ls='dotted', alpha=0.7)
            ax_if.set_xlabel(f"{wcs_object.wcs.ctype[0]} [{wcs_units[0]}]")
            ax_if.set_ylabel(f"{wcs_object.wcs.ctype[1]} [{wcs_units[1]}]")
            cbar_if = fig_if.colorbar(im_if, ax=ax_if, fraction=0.046, pad=0.04)
            cbar_if.set_label(f"{if_var.name}: {if_band_name}")
            plt.suptitle(f"Geometry Data (WCS) - Band '{if_band_name}'\n{os.path.basename(netcdf_file_path)}", y=0.95)
            print("--> OK: IF plot generated.")

        except Exception as e_plot_if: print(f"   ERROR IF plot: {type(e_plot_if).__name__}: {e_plot_if}"); traceback.print_exc()

        # Show all generated plots at the end
        if 'fig_sr' in locals() or 'fig_if' in locals():
             print("\n   Showing plot windows...")
             plt.show()

    else: # WCS not available
        print("\n3. WCS not available. Generating Pixel Index-based Plots...")
        try:
            # Plot SR (Indices)
            sr_var = xr_dataset['spectral_data']
            if not (0 <= spectral_band_index_to_plot < sr_var.sizes['wavelength']): spectral_band_index_to_plot = 0
            data_slice_sr = sr_var.isel(wavelength=spectral_band_index_to_plot).load()
            wavelength_val = data_slice_sr['wavelength'].item(); wl_units = data_slice_sr['wavelength'].attrs.get('units', '?')
            plt.figure(figsize=(8,7)); plt.imshow(data_slice_sr.data, cmap='viridis', origin='lower')
            plt.xlabel("Sample Index"); plt.ylabel("Line Index"); plt.colorbar(label=f"{sr_var.name} @ {wavelength_val:.3f} {wl_units}")
            plt.title(f"Spectral Data (Pixel Indices) - Band {spectral_band_index_to_plot}\n{os.path.basename(netcdf_file_path)}")
            print("--> OK: SR plot (indices) generated.")

            # Plot IF (Indices)
            if_var = xr_dataset['geometry_angles']
            if not (0 <= geometry_band_index_to_plot < if_var.sizes['if_band']): geometry_band_index_to_plot = 0
            data_slice_if = if_var.isel(if_band=geometry_band_index_to_plot).load()
            if_band_name = str(data_slice_if['if_band'].item())
            plt.figure(figsize=(8,7)); plt.imshow(data_slice_if.data, cmap='magma', origin='lower')
            plt.xlabel("Sample Index"); plt.ylabel("Line Index"); plt.colorbar(label=f"{if_var.name}: {if_band_name}")
            plt.title(f"Geometry Data (Pixel Indices) - Band '{if_band_name}'\n{os.path.basename(netcdf_file_path)}")
            print("--> OK: IF plot (indices) generated.")

            print("\n   Showing plot windows (indices)...")
            plt.show()

        except Exception as e_plot_idx: print(f"   ERROR index plot: {e_plot_idx}"); traceback.print_exc()


except FileNotFoundError as e: print(f"\nERROR: File not found - {e}")
except ImportError as e: print(f"\nERROR: Missing library - {e}")
except Exception as e:
    print(f"\n--> UNEXPECTED ERROR in read/usage process:")
    print(f"    Type: {type(e).__name__}, Msg: {e}")
    traceback.print_exc()
finally:
    if xr_dataset is not None:
        try: xr_dataset.close()
        except: pass
    script_duration = time.perf_counter() - script_start_time
    print(f"\n--- Total Read Execution Time: {script_duration:.2f} seconds ---")
    print("\n--- Read/Usage Script Completed ---")