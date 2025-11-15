#!/usr/bin/env python3
"""
Example of astrogea usage with cloud storage (S3).
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
    print("=== Astrogea - Cloud Storage Example ===")
    
    # 1. Cloud configuration
    print("\n1. Cloud configuration...")
    config = create_config(Environment.LOCAL)
    
    # Configure S3 (replace with your values)
    config.storage.s3 = {
        'bucket_name': 'my-astrogea-bucket',  # Replace with your bucket
        'region': 'us-east-1',
        'aws_access_key_id': os.environ.get('AWS_ACCESS_KEY_ID'),
        'aws_secret_access_key': os.environ.get('AWS_SECRET_ACCESS_KEY')
    }
    
    print(f"   S3 Bucket: {config.storage.s3['bucket_name']}")
    print(f"   Region: {config.storage.s3['region']}")
    
    # 2. Storage Manager
    print("\n2. Storage Manager Initialization...")
    storage_manager = create_storage_manager(config.get_storage_config())
    print(f"   Available backends: {config.get_available_storage_backends()}")
    
    # 3. Input files (from S3)
    print("\n3. Input file preparation...")
    input_sr_s3 = "s3://my-astrogea-bucket/input/sr_file.hdr"
    input_if_s3 = "s3://my-astrogea-bucket/input/if_file.hdr"
    output_s3 = "s3://my-astrogea-bucket/results/output.nc"
    
    # Temporary local files
    input_sr_local = "temp_sr.hdr"
    input_if_local = "temp_if.hdr"
    output_local = "temp_output.nc"
    
    print(f"   Input SR: {input_sr_s3}")
    print(f"   Input IF: {input_if_s3}")
    print(f"   Output: {output_s3}")
    
    try:
        # 4. Download files from S3
        print("\n4. Download files from S3...")
        if storage_manager.exists(input_sr_s3):
            storage_manager.download_to_local(input_sr_s3, input_sr_local)
            print(f"   Downloaded: {input_sr_s3}")
        else:
            print(f"   File not found on S3: {input_sr_s3}")
            print("   Using local example file...")
            input_sr_local = "data/frt00006fbd_07_sr164j_mtr3.hdr"
        
        if storage_manager.exists(input_if_s3):
            storage_manager.download_to_local(input_if_s3, input_if_local)
            print(f"   Downloaded: {input_if_s3}")
        else:
            print(f"   File not found on S3: {input_if_s3}")
            print("   Using local example file...")
            input_if_local = "data/frt00006fbd_07_if164j_mtr3.hdr"
        
        # 5. Local processing
        print("\n5. Processing...")
        ds = process_crism_file(
            base_sr_path=input_sr_local.replace('.hdr', ''),
            base_if_path=input_if_local.replace('.hdr', ''),
            output_nc_path=output_local,
            use_dask=True
        )
        
        print("   Processing completed!")
        
        # 6. Upload result to S3
        print("\n6. Upload result to S3...")
        storage_manager.upload_from_local(output_local, output_s3)
        print(f"   Result uploaded to: {output_s3}")
        
        # 7. Cleanup temporary files
        print("\n7. Cleanup temporary files...")
        for temp_file in [input_sr_local, input_if_local, output_local]:
            if os.path.exists(temp_file) and temp_file.startswith('temp_'):
                os.remove(temp_file)
                print(f"   Removed: {temp_file}")
        
    except Exception as e:
        print(f"   Error: {e}")
        return
    
    print("\n=== Completed! ===")
    print(f"Result available at: {output_s3}")

if __name__ == "__main__":
    main()

