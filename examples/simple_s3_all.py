#!/usr/bin/env python3
"""
Example of batch processing all CRISM files from an S3/MinIO bucket.
Processes all .hdr files in the bucket and saves the results.
"""

import os
import argparse
import logging
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add astrogea to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from astrogea.config import create_config, Environment
from astrogea.storage import create_storage_manager
from astrogea.core import envi_to_xarray_wcs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Thread-safe counter for progress
progress_lock = threading.Lock()
processed_count = 0
error_count = 0

def process_single_file(storage_manager, file_path, output_dir, temp_dir, file_index, total_files):
    """
    Process a single CRISM file.
    
    Args:
        storage_manager: Storage manager for S3
        file_path: File path in S3 bucket
        output_dir: Local output directory
        temp_dir: Temporary directory
        file_index: File index (for progress)
        total_files: Total number of files to process
    
    Returns:
        dict: Processing result
    """
    global processed_count, error_count
    
    result = {
        'file': file_path,
        'status': 'pending',
        'error': None,
        'output_file': None,
        'file_size': 0
    }
    
    try:
        # Create temporary directory for this file
        file_temp_dir = temp_dir / f"temp_{file_index}"
        file_temp_dir.mkdir(exist_ok=True)
        
        # Local paths
        file_base = file_path.replace('.hdr', '')
        hdr_local = file_temp_dir / f"{Path(file_base).name}.hdr"
        img_local = file_temp_dir / f"{Path(file_base).name}.img"
        
        # Download .hdr file
        logging.info(f"[{file_index+1}/{total_files}] Download {file_path}...")
        storage_manager.download_to_local(f"s3://{file_path}", str(hdr_local))
        
        # Download .img file (if exists)
        img_s3 = file_path.replace('.hdr', '.img')
        if storage_manager.exists(f"s3://{img_s3}"):
            storage_manager.download_to_local(f"s3://{img_s3}", str(img_local))
        
        # Processing
        logging.info(f"[{file_index+1}/{total_files}] Processing {file_path}...")
        ds = envi_to_xarray_wcs(str(hdr_local))
        
        # Save result
        output_file = output_dir / f"{Path(file_base).name}_processed.nc"
        ds.to_netcdf(str(output_file))
        
        # Update result
        result.update({
            'status': 'success',
            'output_file': str(output_file),
            'file_size': output_file.stat().st_size / (1024*1024)  # MB
        })
        
        # Update thread-safe counters
        with progress_lock:
            processed_count += 1
            logging.info(f"[{processed_count}/{total_files}] Completed: {file_path}")
        
        # Cleanup temporary files
        import shutil
        shutil.rmtree(file_temp_dir)
        
    except Exception as e:
        result.update({
            'status': 'error',
            'error': str(e)
        })
        
        with progress_lock:
            error_count += 1
            logging.error(f"[{file_index+1}/{total_files}] Error {file_path}: {e}")
        
        # Cleanup in case of error
        try:
            if 'file_temp_dir' in locals() and file_temp_dir.exists():
                import shutil
                shutil.rmtree(file_temp_dir)
        except Exception as cleanup_error:
            logging.warning(f"Cleanup failed for {file_path}: {cleanup_error}")
    
    return result

def main():
    logging.info("Astrogea - S3/MinIO Batch Processing")
    
    # 1. S3/MinIO Configuration
    parser = argparse.ArgumentParser(description="Batch process CRISM files from S3/MinIO")
    parser.add_argument("--bucket", default=os.environ.get("AWS_BUCKET", ""), help="S3 bucket name")
    parser.add_argument("--endpoint", default=os.environ.get("AWS_ENDPOINT_URL", ""), help="S3/MinIO endpoint URL")
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"), help="AWS region")
    parser.add_argument("--output", default="output_s3_batch", help="Output directory")
    parser.add_argument("--temp", default="temp_s3_batch", help="Temporary directory")
    parser.add_argument("--threads", type=int, default=4, help="Max parallel threads")
    args = parser.parse_args()
    logging.info("S3/MinIO Configuration...")
    
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
    config.storage.s3 = s3_config['s3']
    config.storage.local = s3_config['local']
    
    logging.info(f"Bucket: {s3_config['s3']['bucket_name']}")
    logging.info(f"Endpoint: {s3_config['s3']['endpoint_url']}")
    
    # 2. Storage Manager
    logging.info("Storage Manager Initialization...")
    try:
        storage_manager = create_storage_manager(config.get_storage_config())
        logging.info(f"Available backends: {config.get_available_storage_backends()}")
    except Exception as e:
        logging.error(f"Unable to initialize storage manager: {e}")
        return
    
    # 3. List all files in bucket
    logging.info("S3 bucket scan...")
    try:
        s3_files = storage_manager.list_files("s3://")
        logging.info(f"Found {len(s3_files)} total files in bucket")
    except Exception as e:
        logging.error(f"Unable to access S3 bucket: {e}")
        return
    
    # 4. Filter CRISM files (.hdr)
    logging.info("Filter CRISM files...")
    crism_files = [f for f in s3_files if f.endswith('.hdr')]
    logging.info(f"Found {len(crism_files)} CRISM files (.hdr):")
    
    for i, file_path in enumerate(crism_files):
        logging.debug(f"{i+1:3d}. {file_path}")
    
    if not crism_files:
        logging.warning("No CRISM files found in bucket")
        return
    
    # 5. Processing configuration
    logging.info("Processing configuration...")
    
    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Temporary directory
    temp_dir = Path(args.temp)
    temp_dir.mkdir(exist_ok=True)
    
    # Parallelism configuration
    max_workers = min(args.threads, len(crism_files))
    use_parallel = len(crism_files) > 1
    
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Temporary directory: {temp_dir}")
    logging.info(f"Files to process: {len(crism_files)}")
    logging.info(f"Parallel processing: {'Yes' if use_parallel else 'No'}")
    if use_parallel:
        logging.info(f"Parallel threads: {max_workers}")
    
    # 6. Batch processing
    logging.info("Starting batch processing...")
    start_time = time.time()
    
    results = []
    
    if use_parallel:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_file = {
                executor.submit(
                    process_single_file, 
                    storage_manager, 
                    file_path, 
                    output_dir, 
                    temp_dir, 
                    i, 
                    len(crism_files)
                ): file_path 
                for i, file_path in enumerate(crism_files)
            }
            
            # Collect results
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({
                        'file': file_path,
                        'status': 'error',
                        'error': str(e)
                    })
                    with progress_lock:
                        error_count += 1
    else:
        # Sequential processing
        for i, file_path in enumerate(crism_files):
            result = process_single_file(
                storage_manager, 
                file_path, 
                output_dir, 
                temp_dir, 
                i, 
                len(crism_files)
            )
            results.append(result)
    
    # 7. Final statistics
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Calculate statistics from results
    successful_count = len([r for r in results if r['status'] == 'success'])
    failed_count = len([r for r in results if r['status'] == 'error'])
    
    logging.info("Processing statistics...")
    logging.info(f"Total time: {processing_time:.2f} seconds")
    logging.info(f"Files processed successfully: {successful_count}")
    logging.info(f"Files with errors: {failed_count}")
    logging.info(f"Total files: {len(crism_files)}")
    
    if successful_count > 0:
        avg_time = processing_time / successful_count
        logging.info(f"Average time per file: {avg_time:.2f} seconds")
    
    # 8. Result details
    logging.info("Result details...")
    
    successful_files = [r for r in results if r['status'] == 'success']
    failed_files = [r for r in results if r['status'] == 'error']
    
    if successful_files:
        logging.info(f"Files processed successfully ({len(successful_files)}):")
        total_size = 0
        for result in successful_files:
            size_mb = result['file_size']
            total_size += size_mb
            logging.info(f"- {result['file']} -> {result['output_file']} ({size_mb:.2f} MB)")
        
        logging.info(f"Total output size: {total_size:.2f} MB")
    
    if failed_files:
        logging.info(f"Files with errors ({len(failed_files)}):")
        for result in failed_files:
            logging.info(f"- {result['file']}: {result['error']}")
    
    # 9. Upload results to S3 (optional)
    logging.info("Upload results to S3...")
    upload_count = 0
    
    for result in successful_files:
        try:
            if result['output_file'] and Path(result['output_file']).exists():
                # Create S3 path for result
                original_name = Path(result['file']).stem
                s3_result_path = f"processed_results/{original_name}_processed.nc"
                
                logging.info(f"Upload {result['output_file']} -> s3://{s3_result_path}")
                storage_manager.upload_from_local(result['output_file'], f"s3://{s3_result_path}")
                upload_count += 1
        except Exception as e:
            logging.error(f"Upload error {result['file']}: {e}")
    
    logging.info(f"{upload_count} files uploaded to S3")
    
    # 10. Final cleanup
    logging.info("Final cleanup...")
    try:
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logging.info("Temporary directory removed")
    except Exception as e:
        logging.warning(f"Unable to remove temporary directory: {e}")
    
    # 11. Final summary
    logging.info("Final Summary")
    logging.info(f"Files processed: {successful_count}/{len(crism_files)}")
    logging.info(f"Success rate: {(successful_count/len(crism_files)*100):.1f}%")
    logging.info(f"Total time: {processing_time:.2f} seconds")
    logging.info(f"Results saved in: {output_dir}")
    logging.info(f"Files uploaded to S3: {upload_count}")
    
    if successful_count > 0:
        logging.info(f"Average time per file: {processing_time/successful_count:.2f} seconds")
    
    logging.info("Completed!")

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
        crism_files = [f for f in files if f.endswith('.hdr')]
        
        logging.info("Connection successful!")
        logging.info(f"Total files in bucket: {len(files)}")
        logging.info(f"CRISM files (.hdr): {len(crism_files)}")
        
        # Show some CRISM files
        logging.info("First 10 CRISM files:")
        for i, file_path in enumerate(crism_files[:10]):
            logging.info(f"{i+1:2d}. {file_path}")
        
        if len(crism_files) > 10:
            logging.info(f"... and {len(crism_files) - 10} more files")
        
        return True
        
    except Exception as e:
        logging.error(f"Connection error: {e}")
        return False

if __name__ == "__main__":
    # First test the connection
    if test_s3_connection():
        logging.info("="*60)
        # If connection works, run batch processing
        main()
    else:
        logging.error("Unable to proceed without valid S3 connection.")
        logging.error("Check: 1) credentials 2) endpoint URL 3) bucket access 4) boto3 installed")
