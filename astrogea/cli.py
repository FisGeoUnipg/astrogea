import argparse
import sys
from .core import process_crism_file
from .config import create_config, Environment
from .orchestrator import create_job_orchestrator
from .distributed import create_distributed_processor

def main():
    parser = argparse.ArgumentParser(description="Astrogea - CRISM data processing with distributed computing.")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process CRISM files')
    process_parser.add_argument("--sr", required=True, help="Base path to the .hdr Surface Reflectance file")
    process_parser.add_argument("--ifile", required=True, help="Base path to the .hdr Incidence Factor file")
    process_parser.add_argument("--out", required=True, help="Output NetCDF file")
    process_parser.add_argument("--dask", action="store_true", help="Use Dask for processing")
    process_parser.add_argument("--env", choices=['local', 'kubernetes'], default='local', help="Environment")
    
    # Cluster command
    cluster_parser = subparsers.add_parser('cluster', help='Manage cluster operations')
    cluster_parser.add_argument("--status", action="store_true", help="Show cluster status")
    cluster_parser.add_argument("--dashboard", action="store_true", help="Open dashboard")
    cluster_parser.add_argument("--scale", type=int, help="Scale workers")
    
    # Job command
    job_parser = subparsers.add_parser('job', help='Manage jobs')
    job_parser.add_argument("--submit", help="Submit job (JSON config)")
    job_parser.add_argument("--status", help="Get job status")
    job_parser.add_argument("--list", action="store_true", help="List jobs")
    job_parser.add_argument("--cancel", help="Cancel job")
    
    args = parser.parse_args()
    
    if args.command == 'process':
        _handle_process(args)
    elif args.command == 'cluster':
        _handle_cluster(args)
    elif args.command == 'job':
        _handle_job(args)
    else:
        parser.print_help()

def _handle_process(args):
    """Handle process command."""
    print(f"Processing CRISM files...")
    print(f"Environment: {args.env}")
    
    ds = process_crism_file(args.sr, args.ifile, args.out, use_dask=args.dask)
    print(f"✅ File saved: {args.out}")
    print(f"Dataset shape: {ds.dims}")

def _handle_cluster(args):
    """Handle cluster command."""
    config = create_config(Environment.KUBERNETES if args.env == 'kubernetes' else Environment.LOCAL)
    
    if args.status:
        from .distributed import create_distributed_processor
        processor = create_distributed_processor(config.get_dask_client_config())
        info = processor.cluster_manager.get_cluster_info()
        print(f"Cluster Status: {info}")
    
    if args.dashboard:
        print("Dashboard available at: http://localhost:8787")
        print("Run: kubectl port-forward svc/dask-scheduler 8787:8787 -n astrogea")
    
    if args.scale:
        print(f"Scaling workers to {args.scale}...")
        # Implementation for scaling

def _handle_job(args):
    """Handle job command."""
    config = create_config(Environment.KUBERNETES)
    orchestrator = create_job_orchestrator({
        'environment': 'kubernetes',
        'storage': config.get_storage_config(),
        'kubernetes': {'namespace': 'astrogea'}
    })
    
    if args.submit:
        import json
        job_config = json.loads(args.submit)
        job_id = orchestrator.submit_job(**job_config)
        print(f"✅ Job submitted: {job_id}")
    
    if args.status:
        status = orchestrator.get_job_status(args.status)
        print(f"Job Status: {status}")
    
    if args.list:
        jobs = orchestrator.list_jobs()
        print(f"Jobs: {len(jobs)}")
        for job in jobs:
            print(f"  {job['job_id']}: {job['status']}")
    
    if args.cancel:
        success = orchestrator.cancel_job(args.cancel)
        print(f"Job cancelled: {success}")
