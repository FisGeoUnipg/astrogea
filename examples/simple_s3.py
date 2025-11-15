#!/usr/bin/env python3
"""
Example of use of astrogee with S3/Minio storage.
Elaborates CRISM files from a Bucket S3 and save the result.
"""

import os
import argparse
import logging
import sys
from pathlib import Path

# Add astrogea to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from astrogea.config import create_config, Environment
from astrogea.storage import create_storage_manager
from astrogea.core import envi_to_xarray_wcs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def main():
    logging.info("Astrogea - S3/MinIO Example")
    
    # 1. S3/MinIO Configuration
    parser = argparse.ArgumentParser(description="Process a CRISM file from S3/MinIO")
    parser.add_argument("--bucket", default=os.environ.get("AWS_BUCKET", ""), help="S3 bucket name")
    parser.add_argument("--endpoint", default=os.environ.get("AWS_ENDPOINT_URL", ""), help="S3/MinIO endpoint URL")
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"), help="AWS region")
    parser.add_argument("--target_file", default="frt00003e12_07_de166l_mtr1.hdr", help="CRISM .hdr to process")
    parser.add_argument("--output_dir", default="output_s3", help="Output directory")
    args = parser.parse_args()
    logging.info("S3/MinIO Configuration...")
    
    # MinIO/AWS Configuration
    s3_config = {
        's3': {
            'bucket_name': args.bucket or os.environ.get('AWS_BUCKET', ''),
            'aws_access_key_id': os.environ.get('AWS_ACCESS_KEY_ID'),
            'aws_secret_access_key': os.environ.get('AWS_SECRET_ACCESS_KEY'),
            'region': args.region,
            'endpoint_url': args.endpoint or os.environ.get('AWS_ENDPOINT_URL')
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
    
    logging.info(f"Bucket: {s3_config['s3']['bucket_name']}")
    logging.info(f"Endpoint: {s3_config['s3']['endpoint_url']}")
    logging.info(f"Environment: {config.environment.value}")
    
    # 2. Storage Manager
    logging.info("Storage Manager Initialization...")
    try:
        storage_manager = create_storage_manager(config.get_storage_config())
        logging.info(f"Available backends: {config.get_available_storage_backends()}")
    except Exception as e:
        logging.error(f"Unable to initialize storage manager: {e}")
        logging.info("Make sure you have boto3 installed: pip install boto3")
        return
    
    # 3. List files in S3 bucket
    logging.info("List files in S3 bucket...")
    try:
        # List all files in the bucket
        s3_files = storage_manager.list_files("s3://")
        logging.info(f"Found {len(s3_files)} files in bucket:")
        
        for i, file_path in enumerate(s3_files[:10]):  # Show only first 10
            logging.debug(f"{i+1}. {file_path}")
        
        if len(s3_files) > 10:
            logging.info(f"... and {len(s3_files) - 10} more files")
            
    except Exception as e:
        logging.error(f"Unable to access S3 bucket: {e}")
        logging.info("Check credentials and endpoint URL")
        return
    
    # 4. Search for CRISM files in bucket
    logging.info("Search for CRISM files...")
    crism_files = []
    
    # Search for .hdr files (header files)
    for file_path in s3_files:
        if file_path.endswith('.hdr'):
            crism_files.append(file_path)
    
    logging.info(f"Found {len(crism_files)} CRISM files (.hdr):")
    for file_path in crism_files:
        logging.debug(f"- {file_path}")
    
    if not crism_files:
        logging.warning("No CRISM files found in bucket")
        logging.info("Make sure there are .hdr files in the bucket")
        return
    
    # 5. Select file to process
    logging.info("Select file to process...")
    
    # Specific file to process
    target_file = args.target_file
    
    # Check if file exists in bucket
    if target_file not in s3_files:
        logging.error(f"File not found in bucket: {target_file}")
        logging.info("Available files:")
        for file_path in crism_files:
            logging.info(f"- {file_path}")
        return
    
    selected_file = target_file
    logging.info(f"Selected file: {selected_file}")
    
    # 6. Download file from S3
    logging.info("Download file from S3...")
    try:
        # Create local directory for temporary files
        local_data_dir = Path("temp_s3_data")
        local_data_dir.mkdir(exist_ok=True)
        
        # Download CRISM file
        file_base = selected_file.replace('.hdr', '')
        hdr_local = local_data_dir / f"{Path(file_base).name}.hdr"
        img_local = local_data_dir / f"{Path(file_base).name}.img"
        
        logging.info(f"Download {selected_file}...")
        storage_manager.download_to_local(f"s3://{selected_file}", str(hdr_local))
        
        # Download image file (if exists)
        img_s3 = selected_file.replace('.hdr', '.img')
        if storage_manager.exists(f"s3://{img_s3}"):
            logging.info(f"Download {img_s3}...")
            storage_manager.download_to_local(f"s3://{img_s3}", str(img_local))
        
        logging.info("Download completed!")
        
    except Exception as e:
        logging.error(f"Error during download: {e}")
        return
    
    # 7. CRISM file processing
    logging.info("CRISM file processing...")
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Local paths for processing
        output_file = output_dir / "result_s3.nc"
        
        logging.info(f"Processing file: {hdr_local}")
        logging.info(f"Output: {output_file}")
        
        # Process single file
        ds = envi_to_xarray_wcs(str(hdr_local))
        
        # Save dataset
        ds.to_netcdf(str(output_file))
        
        logging.info("Processing completed!")
        logging.info(f"Dataset dims: {ds.dims}")
        logging.info(f"Variables: {list(ds.data_vars.keys())}")
        
    except Exception as e:
        logging.error(f"Error during processing: {e}")
        return
    
    # 8. Upload result to S3 (optional)
    logging.info("Upload result to S3 (optional)...")
    try:
        if output_file.exists():
            # Upload result
            result_s3_path = f"results/{output_file.name}"
            logging.info(f"Upload {output_file} -> s3://{result_s3_path}")
            storage_manager.upload_from_local(str(output_file), f"s3://{result_s3_path}")
            logging.info("Upload completed!")
        else:
            logging.info("No output file to upload")
            
    except Exception as e:
        logging.error(f"Error during upload: {e}")
    
    # 9. Cleanup temporary files
    logging.info("Cleanup temporary files...")
    try:
        import shutil
        if local_data_dir.exists():
            shutil.rmtree(local_data_dir)
            logging.info("Temporary files removed")
    except Exception as e:
        logging.warning(f"Unable to remove temporary files: {e}")
    
    # 10. Verify result
    logging.info("Verify result...")
    if output_file.exists():
        file_size = output_file.stat().st_size / (1024*1024)  # MB
        logging.info(f"File created: {output_file}")
        logging.info(f"Size: {file_size:.2f} MB")
    else:
        logging.info(f"File not created: {output_file}")
    
    logging.info("Completed!")
    logging.info("Notes:")
    logging.info("- Credentials are read from environment/CLI")
    logging.info("- Temporary files are downloaded to ./temp_s3_data/ and then removed")
    logging.info("- Result is saved in ./output_s3/result_s3.nc")

def test_s3_connection():
    """Test S3 connection without processing."""
    logging.info("S3 Connection Test")
    
    s3_config = {
        's3': {
            'bucket_name': os.environ.get('AWS_BUCKET', ''),
            'aws_access_key_id': os.environ.get('AWS_ACCESS_KEY_ID'),
            'aws_secret_access_key': os.environ.get('AWS_SECRET_ACCESS_KEY'),
            'region': os.environ.get('AWS_REGION', 'us-east-1'),
            'endpoint_url': os.environ.get('AWS_ENDPOINT_URL')
        }
    }
    
    try:
        storage_manager = create_storage_manager(s3_config)
        
        # Test connection
        files = storage_manager.list_files("s3://")
        logging.info(f"Connection successful! Found {len(files)} files")
        
        # Show some files
        for i, file_path in enumerate(files[:5]):
            logging.info(f"{i+1}. {file_path}")
        
        return True
        
    except Exception as e:
        logging.error(f"Connection error: {e}")
        return False

if __name__ == "__main__":
    # First test the connection
    if test_s3_connection():
        logging.info("="*50)
        # If connection works, run the complete example
        main()
    else:
        logging.error("Unable to proceed without valid S3 connection.")
        logging.error("Check: 1) credentials 2) endpoint URL 3) bucket access 4) boto3 installed")
