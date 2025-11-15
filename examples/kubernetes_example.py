#!/usr/bin/env python3
"""
Example of astrogea usage on Kubernetes.
"""

import os
import sys
import time
from pathlib import Path

# Add astrogea to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from astrogea.config import create_config, Environment
from astrogea.orchestrator import create_job_orchestrator

def main():
    print("=== Astrogea - Kubernetes Example ===")
    
    # 1. Kubernetes configuration
    print("\n1. Kubernetes Configuration...")
    config = create_config(Environment.KUBERNETES)
    
    # Configure cloud storage
    config.storage.s3 = {
        'bucket_name': os.environ.get('ASTROGEA_S3_BUCKET', 'my-astrogea-bucket'),
        'region': os.environ.get('AWS_REGION', 'us-east-1')
    }
    
    print(f"   Environment: {config.environment.value}")
    print(f"   S3 Bucket: {config.storage.s3['bucket_name']}")
    
    # 2. Job Orchestrator
    print("\n2. Job Orchestrator Initialization...")
    orchestrator = create_job_orchestrator({
        'environment': 'kubernetes',
        'storage': config.get_storage_config(),
        'kubernetes': {
            'namespace': 'astrogea',
            'image': 'astrogea:latest'
        },
        'max_concurrent_jobs': 3
    })
    
    # 3. Job Submission
    print("\n3. Job submission...")
    job_id = orchestrator.submit_job(
        job_type="crism",
        input_files=[
            "s3://my-astrogea-bucket/input/sr_file.hdr,s3://my-astrogea-bucket/input/if_file.hdr"
        ],
        output_path="s3://my-astrogea-bucket/results/",
        processing_params={
            "use_dask": True
        },
        priority=1
    )
    
    print(f"   Job ID: {job_id}")
    
    # 4. Job Monitoring
    print("\n4. Job monitoring...")
    max_wait_time = 300  # 5 minutes
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        status = orchestrator.get_job_status(job_id)
        
        if status:
            print(f"   Status: {status['status']}")
            
            if status['status'] == 'completed':
                print("   Job completed!")
                print(f"   Result files: {status.get('result_files', [])}")
                break
            elif status['status'] == 'failed':
                print(f"   Job failed: {status.get('error_message', 'Unknown error')}")
                break
            elif status['status'] == 'running':
                print("   Job running...")
        
        time.sleep(10)  # Check every 10 seconds
    
    else:
        print("   Timeout reached")
    
    # 5. Statistics
    print("\n5. Cluster statistics...")
    stats = orchestrator.get_statistics()
    print(f"   Total jobs: {stats['total_jobs']}")
    print(f"   Active jobs: {stats['active_jobs']}")
    print(f"   Completed jobs: {stats['completed_jobs']}")
    print(f"   Failed jobs: {stats['failed_jobs']}")
    
    print("\n=== Completed! ===")

if __name__ == "__main__":
    main()

