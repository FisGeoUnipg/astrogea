#!/usr/bin/env python3
"""
Example of batch processing all CRISM files from an S3/MinIO bucket.
Processes all .hdr files in the bucket and saves the results.
"""

import os
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
        print(f"   [{file_index+1}/{total_files}] Download {file_path}...")
        storage_manager.download_to_local(f"s3://{file_path}", str(hdr_local))
        
        # Download .img file (if exists)
        img_s3 = file_path.replace('.hdr', '.img')
        if storage_manager.exists(f"s3://{img_s3}"):
            storage_manager.download_to_local(f"s3://{img_s3}", str(img_local))
        
        # Processing
        print(f"   [{file_index+1}/{total_files}] Processing {file_path}...")
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
            print(f"   ✅ [{processed_count}/{total_files}] Completed: {file_path}")
        
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
            print(f"   ❌ [{file_index+1}/{total_files}] Error {file_path}: {e}")
        
        # Cleanup in case of error
        try:
            if 'file_temp_dir' in locals() and file_temp_dir.exists():
                import shutil
                shutil.rmtree(file_temp_dir)
        except:
            pass
    
    return result

def main():
    print("=== Astrogea - S3/MinIO Batch Processing ===")
    
    # 1. S3/MinIO Configuration
    print("\n1. S3/MinIO Configuration...")
    
    s3_config = {
        's3': {
            'bucket_name': 'vamorini-test',
            'aws_access_key_id': 'aws_access_key_id',
            'aws_secret_access_key': 'aws_secret_access_key',
            'region': 'us-east-1',
            'endpoint_url': 'https://minio-api.eagleprojects.cloud'
        },
        'local': {
            'base_path': './data'
        }
    }
    
    # Create configuration
    config = create_config(Environment.LOCAL)
    config.storage.s3 = s3_config['s3']
    config.storage.local = s3_config['local']
    
    print(f"   Bucket: {s3_config['s3']['bucket_name']}")
    print(f"   Endpoint: {s3_config['s3']['endpoint_url']}")
    
    # 2. Storage Manager
    print("\n2. Storage Manager Initialization...")
    try:
        storage_manager = create_storage_manager(config.get_storage_config())
        print(f"   Available backends: {config.get_available_storage_backends()}")
    except Exception as e:
        print(f"   ERROR: Unable to initialize storage manager: {e}")
        return
    
    # 3. List all files in bucket
    print("\n3. S3 bucket scan...")
    try:
        s3_files = storage_manager.list_files("s3://")
        print(f"   Found {len(s3_files)} total files in bucket")
    except Exception as e:
        print(f"   ERROR: Unable to access S3 bucket: {e}")
        return
    
    # 4. Filter CRISM files (.hdr)
    print("\n4. Filter CRISM files...")
    crism_files = [f for f in s3_files if f.endswith('.hdr')]
    print(f"   Found {len(crism_files)} CRISM files (.hdr):")
    
    for i, file_path in enumerate(crism_files):
        print(f"     {i+1:3d}. {file_path}")
    
    if not crism_files:
        print("   No CRISM files found in bucket")
        return
    
    # 5. Processing configuration
    print("\n5. Processing configuration...")
    
    # Output directory
    output_dir = Path("output_s3_batch")
    output_dir.mkdir(exist_ok=True)
    
    # Temporary directory
    temp_dir = Path("temp_s3_batch")
    temp_dir.mkdir(exist_ok=True)
    
    # Parallelism configuration
    max_workers = min(4, len(crism_files))  # Maximum 4 parallel threads
    use_parallel = len(crism_files) > 1
    
    print(f"   Output directory: {output_dir}")
    print(f"   Temporary directory: {temp_dir}")
    print(f"   Files to process: {len(crism_files)}")
    print(f"   Parallel processing: {'Yes' if use_parallel else 'No'}")
    if use_parallel:
        print(f"   Parallel threads: {max_workers}")
    
    # 6. Batch processing
    print("\n6. Starting batch processing...")
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
    
    print("\n7. Processing statistics...")
    print(f"   Total time: {processing_time:.2f} seconds")
    print(f"   Files processed successfully: {successful_count}")
    print(f"   Files with errors: {failed_count}")
    print(f"   Total files: {len(crism_files)}")
    
    if successful_count > 0:
        avg_time = processing_time / successful_count
        print(f"   Average time per file: {avg_time:.2f} seconds")
    
    # 8. Result details
    print("\n8. Result details...")
    
    successful_files = [r for r in results if r['status'] == 'success']
    failed_files = [r for r in results if r['status'] == 'error']
    
    if successful_files:
        print(f"\n   ✅ Files processed successfully ({len(successful_files)}):")
        total_size = 0
        for result in successful_files:
            size_mb = result['file_size']
            total_size += size_mb
            print(f"     - {result['file']} -> {result['output_file']} ({size_mb:.2f} MB)")
        
        print(f"   📊 Total output size: {total_size:.2f} MB")
    
    if failed_files:
        print(f"\n   ❌ Files with errors ({len(failed_files)}):")
        for result in failed_files:
            print(f"     - {result['file']}: {result['error']}")
    
    # 9. Upload results to S3 (optional)
    print("\n9. Upload results to S3...")
    upload_count = 0
    
    for result in successful_files:
        try:
            if result['output_file'] and Path(result['output_file']).exists():
                # Create S3 path for result
                original_name = Path(result['file']).stem
                s3_result_path = f"processed_results/{original_name}_processed.nc"
                
                print(f"   Upload {result['output_file']} -> s3://{s3_result_path}")
                storage_manager.upload_from_local(result['output_file'], f"s3://{s3_result_path}")
                upload_count += 1
        except Exception as e:
            print(f"   ❌ Upload error {result['file']}: {e}")
    
    print(f"   ✅ {upload_count} files uploaded to S3")
    
    # 10. Final cleanup
    print("\n10. Final cleanup...")
    try:
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print("   Temporary directory removed")
    except Exception as e:
        print(f"   Warning: Unable to remove temporary directory: {e}")
    
    # 11. Final summary
    print("\n=== Final Summary ===")
    print(f"📁 Files processed: {successful_count}/{len(crism_files)}")
    print(f"📊 Success rate: {(successful_count/len(crism_files)*100):.1f}%")
    print(f"⏱️  Total time: {processing_time:.2f} seconds")
    print(f"💾 Results saved in: {output_dir}")
    print(f"☁️  Files uploaded to S3: {upload_count}")
    
    if successful_count > 0:
        print(f"🎯 Average time per file: {processing_time/successful_count:.2f} seconds")
    
    print("\n=== Completed! ===")

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
        crism_files = [f for f in files if f.endswith('.hdr')]
        
        print(f"✅ Connection successful!")
        print(f"📁 Total files in bucket: {len(files)}")
        print(f"🔬 CRISM files (.hdr): {len(crism_files)}")
        
        # Show some CRISM files
        print("\nFirst 10 CRISM files:")
        for i, file_path in enumerate(crism_files[:10]):
            print(f"  {i+1:2d}. {file_path}")
        
        if len(crism_files) > 10:
            print(f"     ... and {len(crism_files) - 10} more files")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

if __name__ == "__main__":
    # First test the connection
    if test_s3_connection():
        print("\n" + "="*60)
        # If connection works, run batch processing
        main()
    else:
        print("\nUnable to proceed without valid S3 connection.")
        print("Check:")
        print("1. Correct credentials (access_key and secret_key)")
        print("2. Correct endpoint URL")
        print("3. Bucket 'vamorini-test' exists and is accessible")
        print("4. boto3 installed: pip install boto3")
