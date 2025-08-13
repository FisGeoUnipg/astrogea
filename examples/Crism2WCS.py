# -*- coding: utf-8 -*-
"""
Script to read main CRISM data (_sr_) and auxiliary data (_if_),
combine them into an xarray.Dataset, extract WCS information,
and save everything in a single NetCDF file.

Correct Version: Saves 'has_wcs' flag as integer (1/0) for NetCDF.
"""

# --- Imports ---
import os
import time
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import warnings
import traceback
import re
import ast

# Importa librerie necessarie
try: import spectral; from spectral import open_image; print(f"spectralpy {spectral.__version__}")
except ImportError: print("ERRORE: pip install spectralpy"); exit()
try: from astropy.wcs import WCS; from astropy.io import fits; from astropy import units as u; print("astropy")
except ImportError: print("ERRORE: pip install astropy"); exit()
try: import netCDF4; print("netCDF4"); NETCDF_ENGINE = 'netcdf4'
except ImportError: print("ATTENZIONE: pip install netCDF4 (per salvare)"); NETCDF_ENGINE = None
try: import dask; import dask.array as da; from dask.diagnostics import ProgressBar; print("dask"); USE_DASK = True
except ImportError: print("ATTENZIONE: pip install \"dask[complete]\" (opzionale)"); USE_DASK = False

# --- WCS Helper Functions ---
def parse_envi_map_info_list(map_info_list):
    info = {}
    if not isinstance(map_info_list, list) or len(map_info_list) < 8: return None
    try:
        parts = [str(p).strip() for p in map_info_list]
        info['projection_name'] = parts[0]
        info['ref_x_1based'] = float(parts[1]); info['ref_y_1based'] = float(parts[2])
        info['ref_coord1'] = float(parts[3]); info['ref_coord2'] = float(parts[4])
        info['pixel_size_x'] = float(parts[5]); info['pixel_size_y'] = float(parts[6])
        info['units'] = 'unknown'; info['datum'] = 'unknown'
        for part in reversed(parts[7:]):
             part_lower = part.lower()
             if 'units=' in part_lower:
                 unit_val = part_lower.split('units=')[-1].strip()
                 if 'meter' in unit_val: info['units'] = 'm'
                 elif 'deg' in unit_val: info['units'] = 'deg'
                 else: info['units'] = unit_val
                 break
             elif 'meter' in part_lower: info['units'] = 'm'; break
             elif 'deg' in part_lower: info['units'] = 'deg'; break
        potential_datum_idx = -2 if 'units=' in parts[-1].lower() else 7
        if len(parts) > potential_datum_idx:
            try: float(parts[potential_datum_idx]);
            except ValueError: info['datum'] = parts[potential_datum_idx]
        if info['units'] == 'm': info['ref_easting'], info['ref_northing'] = info['ref_coord1'], info['ref_coord2']
        elif info['units'] == 'deg': info['ref_lon'], info['ref_lat'] = info['ref_coord1'], info['ref_coord2']
        # print(f"    Parsed map info list: {info}") # Riduci verbosità
        return info
    except (ValueError, IndexError, KeyError) as e: print(f"    ERROR parsing 'map info' list: {e}"); return None

def create_wcs_from_parsed_info(parsed_info, shape):
    if parsed_info is None: return None
    lines, samples = shape; units = parsed_info.get('units', 'unknown'); proj_name = parsed_info.get('projection_name', '').lower()
    try:
        w = WCS(naxis=2)
        w.wcs.crpix = [parsed_info['ref_x_1based'], parsed_info['ref_y_1based']]
        w.wcs.cdelt = [parsed_info['pixel_size_x'], -abs(parsed_info['pixel_size_y'])]
        ctype1, ctype2 = '', ''
        units_lower = units.lower()
        if units_lower == 'm' or units_lower == 'meters':
            w.wcs.crval = [parsed_info['ref_easting'], parsed_info['ref_northing']]
            w.wcs.cunit = ['m', 'm']; ctype1, ctype2 = 'XMETR', 'YMETR'
            # print(f"    INFO: WCS CTYPE generici ('XMETR','YMETR') assegnati per unità metri.") # Riduci verbosità
        elif units_lower == 'deg' or units_lower == 'degree' or units_lower == 'degrees':
             w.wcs.crval = [parsed_info['ref_lon'], parsed_info['ref_lat']]
             w.wcs.cunit = ['deg', 'deg']; ctype1, ctype2 = 'OLON', 'OLAT'
             if 'sinusoidal' in proj_name: ctype1+='-SIN'; ctype2+='-SIN'
             elif 'equirectangular' in proj_name or 'plate carree' in proj_name: ctype1+='-CAR'; ctype2+='-CAR'
             elif 'lambert conformal' in proj_name: ctype1+='-LCC'; ctype2+='-LCC'
             elif 'polar stereographic' in proj_name: ctype1+='-STG'; ctype2+='-STG'
             else: print(f"    WARN: Proiezione gradi '{proj_name}' non mappata a suffisso CTYPE FITS.")
        else:
            print(f"   ERROR: Unità '{units}' non gestite."); return None
        w.wcs.ctype = [ctype1, ctype2]
        print(f"--> OK: Oggetto WCS creato (CTYPE={w.wcs.ctype}, CUNIT={w.wcs.cunit})")
        # print(f"    WCS CRVAL: {w.wcs.crval}, CDELT: {w.wcs.cdelt}, CRPIX: {w.wcs.crpix}") # Riduci verbosità
        return w
    except (KeyError, Exception) as e:
        print(f"   ERROR: Creazione WCS fallita - {type(e).__name__}: {e}")
        return None

# --- Spectral Wrapper Class ---
class SpectralArrayWrapper:
    def __init__(self, spectral_image): self._s=spectral_image;self.shape=self._s.shape;self.dtype=self._s.dtype;self.ndim=len(self.shape)
    def __getitem__(self, key): return self._s[key]

# --- Script Configuration ---
script_start_time = time.perf_counter()
base_file_name = "data/frt00006fbd_07_if164j_mtr3"
base_if_file_name = "data/frt00006fbd_07_if164j_mtr3"
hdr_sr_file_path = f"{base_file_name}.hdr"
img_sr_file_path = f"{base_file_name}.img"
hdr_if_file_path = f"{base_if_file_name}.hdr"
img_if_file_path = f"{base_if_file_name}.img"
output_nc_file = f"{base_file_name}_dataset.nc"
print(f"\n--- Input Files ---"); print(f"  SR: {hdr_sr_file_path}"); print(f"  IF: {hdr_if_file_path}")
print(f"--- Output Files ---"); print(f"  NetCDF: {output_nc_file}")
if USE_DASK:
    NUM_WORKERS = 8; SCHEDULER_TYPE = 'threads'
    print(f"\nDask Config: Scheduler='{SCHEDULER_TYPE}', Workers={NUM_WORKERS}")
    try: dask.config.set(scheduler=SCHEDULER_TYPE, num_workers=NUM_WORKERS)
    except Exception as e: print(f"WARN: Dask config error: {e}")
CHUNKS_SR = {'line': 'auto', 'sample': 'auto', 'wavelength': -1}
CHUNKS_IF = {'line': 'auto', 'sample': 'auto', 'if_band': -1}
print(f"SR Chunks Config (Dask): {CHUNKS_SR}"); print(f"IF Chunks Config (Dask): {CHUNKS_IF}")

# --- Check Input Files ---
print("\n--- Checking Input Files ---"); abort = False
for fpath in [hdr_sr_file_path, img_sr_file_path, hdr_if_file_path, img_if_file_path]:
    if not os.path.exists(fpath): print(f"--> ERROR: File not found: {fpath}"); abort = True
    else: print(f"--> OK: Found: {fpath}")
if abort: exit()

# --- Start Process ---
img_sr = None; img_if = None; xr_dataset = None; wcs_object = None; wcs_header_dict = None

try:
    # 1. Open Images
    print("\n1. Opening files..."); print(f"   SR: '{hdr_sr_file_path}'"); img_sr = open_image(hdr_sr_file_path)
    print(f"--> OK: SR opened {img_sr.shape}"); print(f"   IF: '{hdr_if_file_path}'"); img_if = open_image(hdr_if_file_path)
    print(f"--> OK: IF opened {img_if.shape}")
    if img_sr.shape[:2] != img_if.shape[:2]: raise ValueError(f"Shape mismatch SR/IF")
    lines, samples = img_sr.shape[:2]; bands_sr = img_sr.shape[2]; bands_if = img_if.shape[2]
    print(f"   Spatial Dimensions: L={lines}, S={samples}. SR bands:{bands_sr}, IF bands:{bands_if}")

    # 2. Extract Metadata
    print("\n2. Extracting Metadata...")
    wavelengths = np.linspace(1.0, 2.5, bands_sr); wavelength_units = 'micrometers'
    if hasattr(img_sr, 'bands') and img_sr.bands.centers:
         wavelengths = np.array(img_sr.bands.centers, dtype=np.float32); wavelength_units = img_sr.metadata.get('wavelength units', 'unknown')
    else: print("   WARN: SR wavelengths not found.")
    print(f"   SR Wavelengths: {len(wavelengths)}, Unit: {wavelength_units}")
    if_band_names = [f"if_band_{i}" for i in range(bands_if)]
    if hasattr(img_if, 'metadata') and 'band names' in img_if.metadata:
        bn = img_if.metadata['band names']
        if isinstance(bn, list) and len(bn) == bands_if: if_band_names = [re.sub(r'\s*\(.*\)\s*$', '', b).strip() for b in bn]; print(f"   IF Band Names: {if_band_names}")
        else: print("   WARN: IF 'band names' not valid.")
    else: print("   WARN: IF 'band names' not found.")
    meta_sr = {k:str(v) for k,v in img_sr.metadata.items()} if hasattr(img_sr,'metadata') else {}
    meta_if = {k:str(v) for k,v in img_if.metadata.items()} if hasattr(img_if,'metadata') else {}
    print("\n   Attempting to create WCS from SR header...")
    map_info_data = img_sr.metadata.get('map info', None)
    parsed_map_info = None
    if map_info_data and isinstance(map_info_data, list):
        parsed_map_info = parse_envi_map_info_list(map_info_data)
        if parsed_map_info:
             wcs_object = create_wcs_from_parsed_info(parsed_map_info, (lines, samples))
             if wcs_object:
                 try:
                     wcs_header = wcs_object.to_header(relax=True); wcs_header_dict = dict(wcs_header)
                     wcs_header_dict = {k: v for k, v in wcs_header_dict.items() if v is not None}
                     if 'projection_name' in parsed_map_info: wcs_header_dict['PROJNAME'] = parsed_map_info['projection_name']
                     if 'datum' in parsed_map_info: wcs_header_dict['DATUM'] = parsed_map_info['datum']
                     print("--> OK: WCS header generated.")
                 except Exception as e_hdr: print(f"ERROR WCS to header: {e_hdr}"); wcs_object=None; wcs_header_dict=None
    if not wcs_object: print("   WARNING: WCS info not found/valid in SR header.")

    # 3. Prepare Data Backend
    print("\n3. Preparing data backend..."); print("   SR Wrapper..."); wrapped_sr = SpectralArrayWrapper(img_sr)
    print("   IF Wrapper..."); wrapped_if = SpectralArrayWrapper(img_if)
    if USE_DASK:
        print("\n   Using Dask..."); print("   Dask array SR..."); data_sr_backend = da.from_array(wrapped_sr, chunks=CHUNKS_SR, name=f"spectral-{base_file_name}")
        print("   Dask array IF..."); data_if_backend = da.from_array(wrapped_if, chunks=CHUNKS_IF, name=f"geometry-{base_if_file_name}")
        print(f"--> OK: Dask arrays created (lazy).")
    else:
        print("\n   Loading NumPy..."); print("   SR..."); data_sr_backend = np.array(wrapped_sr[:])
        print("   IF..."); data_if_backend = np.array(wrapped_if[:])
        print(f"--> OK: Data loaded into NumPy.")

    # 4. Create xarray Dataset
    print("\n4. Creating xarray Dataset...")
    line_coords = xr.DataArray(np.arange(lines), dims='line', attrs={'long_name': 'Spatial dimension Y', 'units': 'pixel_index'})
    sample_coords = xr.DataArray(np.arange(samples), dims='sample', attrs={'long_name': 'Spatial dimension X', 'units': 'pixel_index'})
    wavelength_coords = xr.DataArray(wavelengths, dims='wavelength', attrs={'long_name': 'Wavelength', 'units': wavelength_units})
    if_band_coords = xr.DataArray(if_band_names, dims='if_band', attrs={'long_name': 'Geometry Band'})
    print("   SR DataArray..."); sr_da = xr.DataArray(data=data_sr_backend, coords={'line': line_coords, 'sample': sample_coords, 'wavelength': wavelength_coords}, dims=['line', 'sample', 'wavelength'], name='spectral_data', attrs={'description': meta_sr.get('description', 'Spectral data'), 'source_file': hdr_sr_file_path})
    print("   IF DataArray..."); if_da = xr.DataArray(data=data_if_backend, coords={'line': line_coords, 'sample': sample_coords, 'if_band': if_band_coords}, dims=['line', 'sample', 'if_band'], name='geometry_angles', attrs={'description': meta_if.get('description', 'Geometry data'), 'source_file': hdr_if_file_path})
    print("   Combining Dataset...")
    global_attrs = {
        'title': f'CRISM Combined Dataset for {base_file_name}', 'source_files_base': f"{base_file_name}, {base_if_file_name}",
        'processing_script': os.path.basename(__file__), 'creation_date': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        'Conventions': 'CF-1.8', 'history': f'{time.strftime("%Y-%m-%d %H:%M:%S")}: Created dataset.' }

    # *** CORRECTION HERE: Save has_wcs flag as INTEGER ***
    if wcs_header_dict:
        print("   Adding WCS to global attributes (header as string, flag as integer)...")
        global_attrs['wcs_header_dict'] = str(wcs_header_dict) # Save header as string
        global_attrs['has_wcs'] = 1 # Use integer 1 for True
        global_attrs['has_wcs_comment'] = "Integer flag: 1 = WCS header present, 0 = no WCS info."
    else:
        global_attrs['has_wcs'] = 0 # Use integer 0 for False
        global_attrs['has_wcs_comment'] = "Integer flag: 1 = WCS header present, 0 = no WCS info."

    xr_dataset = xr.Dataset({'spectral_data': sr_da, 'geometry_angles': if_da}, coords={'line': line_coords, 'sample': sample_coords, 'wavelength': wavelength_coords, 'if_band': if_band_coords}, attrs=global_attrs)
    print("--> OK: xarray Dataset created!"); print("\n--- Dataset Info ---"); print(xr_dataset)

    # 5. Save NetCDF Dataset
    if NETCDF_ENGINE and xr_dataset is not None:
        print(f"\n5. Saving NetCDF Dataset ({output_nc_file})...")
        encoding_options = { 'spectral_data': {'zlib': True, 'complevel': 4}, 'geometry_angles': {'zlib': True, 'complevel': 4} }
        if sr_da.dtype.kind == 'f': encoding_options['spectral_data']['_FillValue'] = np.nan
        if if_da.dtype.kind == 'f': encoding_options['geometry_angles']['_FillValue'] = np.nan
        print(f"   Engine: '{NETCDF_ENGINE}'..."); print(f"   Encoding: {encoding_options}")
        write_job = xr_dataset.to_netcdf(path=output_nc_file, mode='w', engine=NETCDF_ENGINE, encoding=encoding_options, compute=False)
        print("   Starting computation/writing..."); compute_start_time=time.perf_counter()
        if USE_DASK:
            with ProgressBar(): print(f"    Scheduler '{dask.config.get('scheduler')}'..."); write_job.compute()
        else: write_job.compute()
        compute_duration = time.perf_counter()-compute_start_time
        print(f"--> OK: NetCDF Dataset saved. Writing time: {compute_duration:.2f}s")
    elif xr_dataset is None: print("\nNetCDF saving skipped: Dataset not created.")
    else: print("\nNetCDF saving skipped: NetCDF engine not available.")

# --- Error Handling and Script End ---
except FileNotFoundError as e: print(f"\nERROR: File not found - {e}")
except ImportError as e: print(f"\nERROR: Missing library - {e}")
except ValueError as e: print(f"\nERROR: Invalid value - {e}")
except Exception as e:
    print(f"\n--> UNEXPECTED ERROR:"); print(f"    Type: {type(e).__name__}, Msg: {e}"); traceback.print_exc()
finally:
    if img_sr is not None and hasattr(img_sr, 'fid') and img_sr.fid:
        try: img_sr.fid.close()
        except: pass
    if img_if is not None and hasattr(img_if, 'fid') and img_if.fid:
        try: img_if.fid.close()
        except: pass
    if xr_dataset is not None:
        try: xr_dataset.close()
        except: pass
    script_duration = time.perf_counter() - script_start_time
    print(f"\n--- Total Execution Time: {script_duration:.2f} seconds ---")
    print("\n--- Dataset Combination Script Completed ---")