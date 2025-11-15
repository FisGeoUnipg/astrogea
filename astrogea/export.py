"""
Export functions for astrogea data to various formats including ASCII with WCS.
"""

import os
import numpy as np
import xarray as xr
import logging
from typing import Optional, Dict, Any, Union
from pathlib import Path

logger = logging.getLogger(__name__)

def export_to_ascii_wcs(dataset: xr.Dataset, output_path: str, 
                       variable: str = 'spectral_data', 
                       wavelength_idx: Optional[int] = None,
                       include_coordinates: bool = True) -> str:
    """
    Export dataset to ASCII format with WCS coordinates.
    
    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset to export
    output_path : str
        Output file path
    variable : str
        Variable to export (default: 'spectral_data')
    wavelength_idx : int, optional
        Specific wavelength index to export (if None, exports all)
    include_coordinates : bool
        Include geographic coordinates in output
    
    Returns
    -------
    str
        Path to exported file
    """
    
    if variable not in dataset.data_vars:
        raise ValueError(f"Variable '{variable}' not found in dataset")
    
    data_var = dataset[variable]
    
    # Create output directory
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Exporting {variable} to ASCII WCS format: {output_path}")
    
    with open(output_file, 'w') as f:
        # Write header
        f.write("# Astrogea ASCII WCS Export\n")
        f.write(f"# Variable: {variable}\n")
        f.write(f"# Dataset shape: {data_var.shape}\n")
        f.write(f"# Export time: {np.datetime64('now')}\n")
        
        # Write WCS information if available
        if dataset.attrs.get('has_wcs', 0) == 1:
            f.write("# WCS Information Available\n")
            wcs_header = dataset.attrs.get('wcs_header_dict', '')
            f.write(f"# WCS Header: {wcs_header}\n")
        else:
            f.write("# No WCS Information Available\n")
        
        # Write coordinate information
        f.write("# Coordinate Information:\n")
        for coord_name, coord_data in dataset.coords.items():
            f.write(f"#   {coord_name}: {coord_data.shape} - {coord_data.attrs.get('long_name', 'N/A')} ({coord_data.attrs.get('units', 'N/A')})\n")
        
        f.write("#\n")
        
        # Determine what to export
        if wavelength_idx is not None:
            # Export specific wavelength
            if 'wavelength' in data_var.dims:
                data_slice = data_var.isel(wavelength=wavelength_idx)
                wavelength_val = dataset.wavelength[wavelength_idx].values
                f.write(f"# Wavelength: {wavelength_val} {dataset.wavelength.attrs.get('units', 'N/A')}\n")
            else:
                raise ValueError("Dataset does not have wavelength dimension")
        else:
            # Export all wavelengths
            data_slice = data_var
        
        # Write data
        if include_coordinates and dataset.attrs.get('has_wcs', 0) == 1:
            # Export with geographic coordinates
            _write_ascii_with_coordinates(f, data_slice, dataset)
        else:
            # Export with pixel coordinates
            _write_ascii_pixel_coordinates(f, data_slice, dataset)
    
    logger.info(f"ASCII WCS export completed: {output_path}")
    return str(output_file)

def _write_ascii_with_coordinates(f, data_slice: xr.DataArray, dataset: xr.Dataset):
    """Write ASCII data with geographic coordinates."""
    
    # Get coordinate arrays
    if 'line' in data_slice.dims and 'sample' in data_slice.dims:
        lines = dataset.line.values
        samples = dataset.sample.values
        
        # Write header for geographic coordinates
        f.write("# Format: line sample latitude longitude value\n")
        f.write("# Units: pixel pixel degrees degrees data_units\n")
        f.write("#\n")
        
        # Convert pixel coordinates to geographic (simplified)
        # In a real implementation, you'd use the WCS header to do proper conversion
        for i, line in enumerate(lines):
            for j, sample in enumerate(samples):
                if data_slice.ndim == 2:
                    value = data_slice.values[i, j]
                else:
                    value = data_slice.values[i, j, :]  # Multiple wavelengths
                
                # Simplified coordinate conversion (replace with proper WCS)
                lat = 0.0 + i * 0.001  # Placeholder
                lon = 0.0 + j * 0.001  # Placeholder
                
                if isinstance(value, np.ndarray):
                    # Multiple wavelengths
                    for k, val in enumerate(value):
                        f.write(f"{line} {sample} {lat:.6f} {lon:.6f} {val:.6f}\n")
                else:
                    # Single value
                    f.write(f"{line} {sample} {lat:.6f} {lon:.6f} {value:.6f}\n")
    else:
        # Fallback to pixel coordinates
        _write_ascii_pixel_coordinates(f, data_slice, dataset)

def _write_ascii_pixel_coordinates(f, data_slice: xr.DataArray, dataset: xr.Dataset):
    """Write ASCII data with pixel coordinates."""
    
    # Write header
    f.write("# Format: line sample value\n")
    f.write("# Units: pixel pixel data_units\n")
    f.write("#\n")
    
    # Get coordinate arrays
    if 'line' in data_slice.dims and 'sample' in data_slice.dims:
        lines = dataset.line.values
        samples = dataset.sample.values
        
        for i, line in enumerate(lines):
            for j, sample in enumerate(samples):
                if data_slice.ndim == 2:
                    value = data_slice.values[i, j]
                    f.write(f"{line} {sample} {value:.6f}\n")
                else:
                    # Multiple wavelengths
                    values = data_slice.values[i, j, :]
                    for k, val in enumerate(values):
                        f.write(f"{line} {sample} {k} {val:.6f}\n")
    else:
        # Simple array export
        for i in range(data_slice.size):
            f.write(f"{i} {data_slice.values.flat[i]:.6f}\n")

def export_to_ascii_spectral(dataset: xr.Dataset, output_path: str,
                            variable: str = 'spectral_data',
                            pixel_coords: Optional[tuple] = None) -> str:
    """
    Export spectral data for specific pixel(s) to ASCII format.
    
    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset to export
    output_path : str
        Output file path
    variable : str
        Variable to export
    pixel_coords : tuple, optional
        (line, sample) coordinates for specific pixel
    
    Returns
    -------
    str
        Path to exported file
    """
    
    if variable not in dataset.data_vars:
        raise ValueError(f"Variable '{variable}' not found in dataset")
    
    data_var = dataset[variable]
    
    # Create output directory
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Exporting spectral data to ASCII: {output_path}")
    
    with open(output_file, 'w') as f:
        # Write header
        f.write("# Astrogea Spectral ASCII Export\n")
        f.write(f"# Variable: {variable}\n")
        f.write(f"# Dataset shape: {data_var.shape}\n")
        
        # Write wavelength information
        if 'wavelength' in dataset.coords:
            wavelengths = dataset.wavelength.values
            wavelength_units = dataset.wavelength.attrs.get('units', 'N/A')
            f.write(f"# Wavelength units: {wavelength_units}\n")
            f.write(f"# Number of wavelengths: {len(wavelengths)}\n")
            f.write("#\n")
            f.write("# Format: wavelength value\n")
            f.write(f"# Units: {wavelength_units} data_units\n")
            f.write("#\n")
            
            if pixel_coords is not None:
                # Export specific pixel
                line, sample = pixel_coords
                f.write(f"# Pixel coordinates: line={line}, sample={sample}\n")
                
                if data_var.ndim == 3:  # (line, sample, wavelength)
                    spectrum = data_var.isel(line=line, sample=sample).values
                else:
                    raise ValueError("Dataset must have 3 dimensions for pixel export")
                
                # Write spectrum
                for wavelength, value in zip(wavelengths, spectrum):
                    f.write(f"{wavelength:.6f} {value:.6f}\n")
            else:
                # Export all pixels
                f.write("# Format: line sample wavelength value\n")
                f.write("#\n")
                
                for i in range(data_var.shape[0]):  # lines
                    for j in range(data_var.shape[1]):  # samples
                        for k, wavelength in enumerate(wavelengths):
                            value = data_var.values[i, j, k]
                            f.write(f"{i} {j} {wavelength:.6f} {value:.6f}\n")
        else:
            raise ValueError("Dataset must have wavelength coordinate")
    
    logger.info(f"Spectral ASCII export completed: {output_path}")
    return str(output_file)

def export_to_csv_wcs(dataset: xr.Dataset, output_path: str,
                     variable: str = 'spectral_data',
                     include_metadata: bool = True) -> str:
    """
    Export dataset to CSV format with WCS information.
    
    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset to export
    output_path : str
        Output file path
    variable : str
        Variable to export
    include_metadata : bool
        Include metadata in CSV header
    
    Returns
    -------
    str
        Path to exported file
    """
    
    import pandas as pd
    
    if variable not in dataset.data_vars:
        raise ValueError(f"Variable '{variable}' not found in dataset")
    
    data_var = dataset[variable]
    
    # Create output directory
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Exporting {variable} to CSV WCS format: {output_path}")
    
    # Convert to DataFrame
    if data_var.ndim == 3:  # (line, sample, wavelength)
        # Flatten to 2D for CSV
        data_flat = data_var.values.reshape(-1, data_var.shape[2])
        
        # Create index arrays
        lines = np.repeat(np.arange(data_var.shape[0]), data_var.shape[1])
        samples = np.tile(np.arange(data_var.shape[1]), data_var.shape[0])
        
        # Create DataFrame
        df_data = {
            'line': lines,
            'sample': samples
        }
        
        # Add wavelength columns
        wavelengths = dataset.wavelength.values
        for i, wl in enumerate(wavelengths):
            df_data[f'wavelength_{wl:.3f}'] = data_flat[:, i]
        
        df = pd.DataFrame(df_data)
        
    elif data_var.ndim == 2:  # (line, sample)
        # 2D data
        lines, samples = np.meshgrid(
            np.arange(data_var.shape[0]),
            np.arange(data_var.shape[1]),
            indexing='ij'
        )
        
        df = pd.DataFrame({
            'line': lines.flatten(),
            'sample': samples.flatten(),
            'value': data_var.values.flatten()
        })
    
    else:
        raise ValueError(f"Unsupported data dimensions: {data_var.ndim}")
    
    # Write CSV with metadata
    with open(output_file, 'w') as f:
        if include_metadata:
            # Write metadata header
            f.write("# Astrogea CSV WCS Export\n")
            f.write(f"# Variable: {variable}\n")
            f.write(f"# Dataset shape: {data_var.shape}\n")
            f.write(f"# Export time: {np.datetime64('now')}\n")
            
            if dataset.attrs.get('has_wcs', 0) == 1:
                f.write("# WCS Information Available\n")
                wcs_header = dataset.attrs.get('wcs_header_dict', '')
                f.write(f"# WCS Header: {wcs_header}\n")
            else:
                f.write("# No WCS Information Available\n")
            
            f.write("#\n")
        
        # Write DataFrame
        df.to_csv(f, index=False)
    
    logger.info(f"CSV WCS export completed: {output_path}")
    return str(output_file)

def export_to_geotiff_wcs(dataset: xr.Dataset, output_path: str,
                         variable: str = 'spectral_data',
                         wavelength_idx: int = 0) -> str:
    """
    Export dataset to GeoTIFF format with WCS coordinates.
    
    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset to export
    output_path : str
        Output file path
    variable : str
        Variable to export
    wavelength_idx : int
        Wavelength index to export
    
    Returns
    -------
    str
        Path to exported file
    """
    
    try:
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.crs import CRS
    except ImportError:
        raise ImportError("rasterio is required for GeoTIFF export. Install with: pip install rasterio")
    
    if variable not in dataset.data_vars:
        raise ValueError(f"Variable '{variable}' not found in dataset")
    
    data_var = dataset[variable]
    
    # Create output directory
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Exporting {variable} to GeoTIFF WCS format: {output_path}")
    
    # Get data slice
    if data_var.ndim == 3:  # (line, sample, wavelength)
        data_slice = data_var.isel(wavelength=wavelength_idx)
        wavelength_val = dataset.wavelength[wavelength_idx].values
    elif data_var.ndim == 2:  # (line, sample)
        data_slice = data_var
        wavelength_val = None
    else:
        raise ValueError(f"Unsupported data dimensions: {data_var.ndim}")
    
    # Get spatial dimensions
    height, width = data_slice.shape
    
    # Create transform (simplified - replace with proper WCS)
    if dataset.attrs.get('has_wcs', 0) == 1:
        # Use WCS information if available
        # This is a simplified implementation
        transform = from_bounds(0, 0, width, height, width, height)
        crs = CRS.from_epsg(4326)  # WGS84
    else:
        # Use pixel coordinates
        transform = from_bounds(0, 0, width, height, width, height)
        crs = None
    
    # Write GeoTIFF
    with rasterio.open(
        output_file,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=data_slice.dtype,
        crs=crs,
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(data_slice.values, 1)
        
        # Add metadata
        dst.update_tags(
            title=f"Astrogea Export - {variable}",
            variable=variable,
            wavelength=str(wavelength_val) if wavelength_val is not None else "N/A",
            has_wcs=str(dataset.attrs.get('has_wcs', 0)),
            wcs_header=dataset.attrs.get('wcs_header_dict', 'N/A')
        )
    
    logger.info(f"GeoTIFF WCS export completed: {output_path}")
    return str(output_file)






















