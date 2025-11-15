#!/usr/bin/env python3
"""
Check whether the NetCDF file contains WCS information.
"""

import os
import sys
from pathlib import Path
import xarray as xr

# Aggiungi astrogea al path
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_wcs_info(file_path):
    """Check WCS information in the NetCDF file."""
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    print(f"=== WCS Analysis: {file_path} ===")
    
    try:
        # Load dataset
        ds = xr.open_dataset(file_path)
        
        print(f"\nDataset Information:")
        print(f"   Dimensions: {ds.dims}")
        print(f"   Variables: {list(ds.data_vars.keys())}")
        print(f"   Coordinates: {list(ds.coords.keys())}")
        
        print(f"\nWCS Information:")
        
        # Check WCS flag
        has_wcs = ds.attrs.get('has_wcs', 0)
        print(f"   Flag WCS: {has_wcs}")
        
        if has_wcs == 1:
            print("   WCS disponibile!")
            
            # Show WCS header
            wcs_header = ds.attrs.get('wcs_header_dict', 'N/A')
            print(f"   WCS Header: {wcs_header}")
            
            # Show comment
            wcs_comment = ds.attrs.get('has_wcs_comment', 'N/A')
            print(f"   Commento: {wcs_comment}")
            
        else:
            print("   WCS not available")
            wcs_comment = ds.attrs.get('has_wcs_comment', 'N/A')
            print(f"   Reason: {wcs_comment}")
        
        print(f"\nAll Attributes:")
        for key, value in ds.attrs.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"   {key}: {value[:100]}...")
            else:
                print(f"   {key}: {value}")
        
        # Coordinates information
        print(f"\nAvailable Coordinates:")
        for coord_name, coord_data in ds.coords.items():
            print(f"   {coord_name}:")
            print(f"     - Shape: {coord_data.shape}")
            print(f"     - Range: {coord_data.min().values:.3f} - {coord_data.max().values:.3f}")
            if hasattr(coord_data, 'attrs'):
                print(f"     - Units: {coord_data.attrs.get('units', 'N/A')}")
                print(f"     - Long name: {coord_data.attrs.get('long_name', 'N/A')}")
        
        ds.close()
        
    except Exception as e:
        print(f"Error during analysis: {e}")

def main():
    print("=== WCS Check in NetCDF File ===")
    
    # Check output file of the example
    output_file = "output/result.nc"
    
    if os.path.exists(output_file):
        check_wcs_info(output_file)
    else:
        print(f"Output file not found: {output_file}")
        print("Run first: python examples/simple_example.py")
        
        # Check for other NetCDF files
        nc_files = list(Path(".").glob("**/*.nc"))
        if nc_files:
            print(f"\nFound NetCDF files:")
            for nc_file in nc_files:
                print(f"   {nc_file}")
                check_wcs_info(str(nc_file))
                print()

if __name__ == "__main__":
    main()


















