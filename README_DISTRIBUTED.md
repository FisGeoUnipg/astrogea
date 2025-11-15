# Astrogea - Distributed CRISM Processing

## Quick Start

### Local Processing
```bash
# Setup
./scripts/setup_simple.sh
source venv/bin/activate

# Process files
python examples/simple_example.py
astrogea process --sr data/sr.hdr --ifile data/if.hdr --out output.nc --dask
```

### Kubernetes Processing
```bash
# Deploy cluster
./scripts/deploy-kubernetes.sh

# Submit job
astrogea job --submit '{"job_type":"crism","input_files":["s3://bucket/file.hdr"],"output_path":"s3://results/"}'

# Check status
astrogea cluster --status
astrogea job --list
```

### Cloud Storage
```bash
# S3
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
python examples/cloud_example.py

# Export ASCII WCS
python examples/export_ascii_wcs.py
```

## Features
- Local & Kubernetes processing
- S3, GCS, Azure Blob storage
- Dask distributed computing
- Job orchestration
- ASCII WCS export
- NetCDF with georeferencing

## Commands
- `astrogea process` - Process CRISM files
- `astrogea cluster` - Manage cluster
- `astrogea job` - Manage jobs
- `astrogea cluster --dashboard` - Open dashboard

**Ready to use!**






