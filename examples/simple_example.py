#!/usr/bin/env python3
"""
Minimal example of astrogea usage.
Processes a CRISM file and saves the result.
"""

import os
import sys
from pathlib import Path

# Add astrogea to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from astrogea.config import create_config, Environment
from astrogea.storage import create_storage_manager
from astrogea.core import process_crism_file

def main():
    print("=== Astrogea - Minimal Example ===")
    
    # 1. Configuration
    print("\n1. Configuration...")
    config = create_config(Environment.LOCAL)
    print(f"   Environment: {config.environment.value}")
    
    # 2. Storage Manager
    print("\n2. Storage Initialization...")
    storage_manager = create_storage_manager(config.get_storage_config())
    print(f"   Available backends: {config.get_available_storage_backends()}")
    
    # 3. Input files (use example files)
    print("\n3. Input file preparation...")
    input_sr = "data/frt00006fbd_07_sr164j_mtr3.hdr"
    input_if = "data/frt00006fbd_07_if164j_mtr3.hdr"
    output_file = "output/result.nc"
    
    # Check if files exist
    if not os.path.exists(input_sr):
        print(f"   ERROR: SR file not found: {input_sr}")
        print("   Make sure you have example files in the data/ directory")
        return
    
    if not os.path.exists(input_if):
        print(f"   ERROR: IF file not found: {input_if}")
        return
    
    print(f"   SR file: {input_sr}")
    print(f"   IF file: {input_if}")
    print(f"   Output: {output_file}")
    
    # 4. Processing
    print("\n4. CRISM file processing...")
    try:
        # Create output directory
        os.makedirs("output", exist_ok=True)
        
        # Process files
        ds = process_crism_file(
            base_sr_path=input_sr.replace('.hdr', ''),
            base_if_path=input_if.replace('.hdr', ''),
            output_nc_path=output_file,
            use_dask=True  # Use Dask for parallel processing
        )
        
        print("   ✅ Processing completed!")
        print(f"   Dataset shape: {ds.dims}")
        print(f"   Variables: {list(ds.data_vars.keys())}")
        
    except Exception as e:
        print(f"   Error during processing: {e}")
        return
    
    # 5. Verify result
    print("\n5. Result verification...")
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file) / (1024*1024)  # MB
        print(f"    File created: {output_file}")
        print(f"    Size: {file_size:.2f} MB")
    else:
        print(f"   File not created: {output_file}")
    
    print("\n=== Completed! ===")

if __name__ == "__main__":
    main()




