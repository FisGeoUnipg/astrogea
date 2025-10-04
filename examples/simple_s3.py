#!/usr/bin/env python3
"""
Example of use of astrogee with S3/Minio storage.
Elaborates CRISM files from a Bucket S3 and save the result.
"""

import os
import sys
from pathlib import Path

# Add astrogea to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from astrogea.config import create_config, Environment
from astrogea.storage import create_storage_manager
from astrogea.core import envi_to_xarray_wcs

def main():
    print("=== Astrogea - S3/MinIO Example ===")
    
    # 1. S3/MinIO Configuration
    print("\n1. S3/MinIO Configuration...")
    
    # MinIO Configuration
    s3_config = {
        's3': {
            'bucket_name': 'vamorini-test',
            'aws_access_key_id': 'aws_access_key_id',  # Replace with your access key
            'aws_secret_access_key': 'aws_secret_access_key',  # Replace with your secret key
            'region': 'us-east-1',  # MinIO doesn't use regions, but boto3 requires it
            'endpoint_url': 'https://minio-api.eagleprojects.cloud'
        },
        'local': {
            'base_path': './data'
        }
    }
    
    # Create configuration
    config = create_config(Environment.LOCAL)
    
    # Override storage configuration with S3 config
    config.storage.s3 = s3_config['s3']
    config.storage.local = s3_config['local']
    
    print(f"   Bucket: {s3_config['s3']['bucket_name']}")
    print(f"   Endpoint: {s3_config['s3']['endpoint_url']}")
    print(f"   Environment: {config.environment.value}")
    
    # 2. Storage Manager
    print("\n2. Storage Manager Initialization...")
    try:
        storage_manager = create_storage_manager(config.get_storage_config())
        print(f"   Available backends: {config.get_available_storage_backends()}")
    except Exception as e:
        print(f"   ERROR: Unable to initialize storage manager: {e}")
        print("   Make sure you have boto3 installed: pip install boto3")
        return
    
    # 3. List files in S3 bucket
    print("\n3. List files in S3 bucket...")
    try:
        # List all files in the bucket
        s3_files = storage_manager.list_files("s3://")
        print(f"   Found {len(s3_files)} files in bucket:")
        
        for i, file_path in enumerate(s3_files[:10]):  # Show only first 10
            print(f"     {i+1}. {file_path}")
        
        if len(s3_files) > 10:
            print(f"     ... and {len(s3_files) - 10} more files")
            
    except Exception as e:
        print(f"   ERROR: Unable to access S3 bucket: {e}")
        print("   Check credentials and endpoint URL")
        return
    
    # 4. Search for CRISM files in bucket
    print("\n4. Search for CRISM files...")
    crism_files = []
    
    # Search for .hdr files (header files)
    for file_path in s3_files:
        if file_path.endswith('.hdr'):
            crism_files.append(file_path)
    
    print(f"   Found {len(crism_files)} CRISM files (.hdr):")
    for file_path in crism_files:
        print(f"     - {file_path}")
    
    if not crism_files:
        print("   No CRISM files found in bucket")
        print("   Make sure there are .hdr files in the bucket")
        return
    
    # 5. Select file to process
    print("\n5. Select file to process...")
    
    # Specific file to process
    target_file = "frt00003e12_07_de166l_mtr1.hdr"
    
    # Check if file exists in bucket
    if target_file not in s3_files:
        print(f"   ERROR: File not found in bucket: {target_file}")
        print("   Available files:")
        for file_path in crism_files:
            print(f"     - {file_path}")
        return
    
    selected_file = target_file
    print(f"   Selected file: {selected_file}")
    
    # 6. Download file from S3
    print("\n6. Download file from S3...")
    try:
        # Create local directory for temporary files
        local_data_dir = Path("temp_s3_data")
        local_data_dir.mkdir(exist_ok=True)
        
        # Download CRISM file
        file_base = selected_file.replace('.hdr', '')
        hdr_local = local_data_dir / f"{Path(file_base).name}.hdr"
        img_local = local_data_dir / f"{Path(file_base).name}.img"
        
        print(f"   Download {selected_file}...")
        storage_manager.download_to_local(f"s3://{selected_file}", str(hdr_local))
        
        # Download image file (if exists)
        img_s3 = selected_file.replace('.hdr', '.img')
        if storage_manager.exists(f"s3://{img_s3}"):
            print(f"   Download {img_s3}...")
            storage_manager.download_to_local(f"s3://{img_s3}", str(img_local))
        
        print("   ✅ Download completed!")
        
    except Exception as e:
        print(f"   ERROR during download: {e}")
        return
    
    # 7. CRISM file processing
    print("\n7. CRISM file processing...")
    try:
        # Create output directory
        output_dir = Path("output_s3")
        output_dir.mkdir(exist_ok=True)
        
        # Local paths for processing
        output_file = output_dir / "result_s3.nc"
        
        print(f"   Processing file: {hdr_local}")
        print(f"   Output: {output_file}")
        
        # Process single file
        ds = envi_to_xarray_wcs(str(hdr_local))
        
        # Save dataset
        ds.to_netcdf(str(output_file))
        
        print("   ✅ Processing completed!")
        print(f"   Dataset shape: {ds.dims}")
        print(f"   Variables: {list(ds.data_vars.keys())}")
        
    except Exception as e:
        print(f"   ERROR during processing: {e}")
        return
    
    # 8. Upload result to S3 (optional)
    print("\n8. Upload result to S3...")
    try:
        if output_file.exists():
            # Upload result
            result_s3_path = f"results/{output_file.name}"
            print(f"   Upload {output_file} -> s3://{result_s3_path}")
            storage_manager.upload_from_local(str(output_file), f"s3://{result_s3_path}")
            print("   ✅ Upload completed!")
        else:
            print("   No output file to upload")
            
    except Exception as e:
        print(f"   ERROR during upload: {e}")
    
    # 9. Cleanup temporary files
    print("\n9. Cleanup temporary files...")
    try:
        import shutil
        if local_data_dir.exists():
            shutil.rmtree(local_data_dir)
            print("   Temporary files removed")
    except Exception as e:
        print(f"   Warning: Unable to remove temporary files: {e}")
    
    # 10. Verify result
    print("\n10. Verify result...")
    if output_file.exists():
        file_size = output_file.stat().st_size / (1024*1024)  # MB
        print(f"   File created: {output_file}")
        print(f"   Size: {file_size:.2f} MB")
    else:
        print(f"   File not created: {output_file}")
    
    print("\n=== Completed! ===")
    print("\nNotes:")
    print("- Credentials are already configured in the code")
    print("- Processed file: frt00003e12_07_de166l_mtr1.hdr")
    print("- Temporary files are downloaded to ./temp_s3_data/ and then removed")
    print("- Result is saved in ./output_s3/result_s3.nc")

def test_s3_connection():
    """Test S3 connection without processing."""
    print("=== S3 Connection Test ===")
    
    s3_config = {
        's3': {
            'bucket_name': 'vamorini-test',
            'aws_access_key_id': '1ka7a5gUF5ZaNtHIBd7X',
            'aws_secret_access_key': 'PVEA7bXbUaFDq6X69xHkFX82k6CB5wgIDUWCT3i1',
            'region': 'us-east-1',
            'endpoint_url': 'https://minio-api.eagleprojects.cloud'
        }
    }
    
    try:
        storage_manager = create_storage_manager(s3_config)
        
        # Test connection
        files = storage_manager.list_files("s3://")
        print(f"Connection successful! Found {len(files)} files")
        
        # Show some files
        for i, file_path in enumerate(files[:5]):
            print(f"  {i+1}. {file_path}")
        
        return True
        
    except Exception as e:
        print(f"Connection error: {e}")
        return False

if __name__ == "__main__":
    # First test the connection
    if test_s3_connection():
        print("\n" + "="*50)
        # If connection works, run the complete example
        main()
    else:
        print("\nUnable to proceed without valid S3 connection.")
        print("Check:")
        print("1. Correct credentials (access_key and secret_key)")
        print("2. Correct endpoint URL")
        print("3. Bucket 'vamorini-test' exists and is accessible")
        print("4. boto3 installed: pip install boto3")
