# AstroGea

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/badge/pypi-astrogea-blue.svg)](https://pypi.org/project/astrogea/)

**AstroGea** is an advanced Python library for processing and analyzing CRISM (Compact Reconnaissance Imaging Spectrometer for Mars) spectral data with full support for WCS georeferencing and NetCDF formats. It includes tools for mafic band analysis, distributed processing, and cloud storage.

## 🚀 Key Features

- **CRISM Processing**: Conversion of ENVI files (.hdr/.img) to georeferenced NetCDF datasets
- **Spectral Analysis**: Continuum removal, smoothing, spectral coregistration
- **Mafic Analysis**: Extraction of mafic band parameters (minimum, center, depth, area, asymmetry)
- **Multi-Cloud Storage**: Support for S3, Google Cloud Storage, Azure Blob Storage
- **Distributed Processing**: Dask support for parallel processing
- **Georeferencing**: Complete WCS integration for geographic coordinates
- **Multi-Format Export**: NetCDF, CSV, GeoTIFF, ASCII

## 📦 Installation

### Basic Installation
```bash
pip install astrogea
```

### Complete Installation (with all dependencies)
```bash
pip install astrogea[all]
```

### Installation from Source
```bash
git clone https://github.com/tuutente/astrogea.git
cd astrogea
pip install -e .
```

## 🔧 Dependencies

### Core Dependencies
- `numpy>=1.20.0` - Numerical computations
- `xarray>=2022.1.0` - Multidimensional datasets
- `spectral>=0.22.0` - Spectral data processing
- `astropy>=5.0.0` - Astronomy and WCS
- `scipy>=1.7.0` - Scientific algorithms
- `matplotlib>=3.0.0` - Visualization

### Optional Dependencies
- `dask[array]>=2022.1.0` - Parallel processing
- `netCDF4>=1.6.0` - NetCDF4 support
- `rasterio>=1.2.0` - Geospatial I/O
- `boto3` - AWS S3 support
- `google-cloud-storage` - Google Cloud support
- `azure-storage-blob` - Azure support

## 📚 Quick Start Guide

### Basic CRISM File Processing

```python
from astrogea.core import process_crism_file

# Process CRISM files (SR and IF)
ds = process_crism_file(
    base_sr_path="data/frt00006fbd_07_sr164j_mtr3",
    base_if_path="data/frt00006fbd_07_if164j_mtr3", 
    output_nc_path="output/result.nc",
    use_dask=True
)

print(f"Dataset shape: {ds.dims}")
print(f"Variables: {list(ds.data_vars.keys())}")
```

### Single ENVI File Processing

```python
from astrogea.core import envi_to_xarray_wcs

# Convert ENVI file to xarray with WCS
ds = envi_to_xarray_wcs("data/frt00003e12_07_de166l_mtr1.hdr")
ds.to_netcdf("output/processed.nc")
```

### Spectral Analysis

```python
from astrogea.core import continuum_removal, band_parameters_mafic
import numpy as np

# Example data
img = np.random.rand(5, 5, 20) * 1000  # (Y, X, bands)
wavelength = np.linspace(1000, 2500, 20)
MIN, MAX = 1200, 2200

# Continuum removal
result, x = continuum_removal(img, wavelength, MIN, MAX)

# Mafic band analysis
mafic_map = band_parameters_mafic(result, x, nbands=5)
```

### Cloud Storage

```python
from astrogea.storage import create_storage_manager

# S3 Configuration
s3_config = {
    's3': {
        'bucket_name': 'my-bucket',
        'aws_access_key_id': 'your-key',
        'aws_secret_access_key': 'your-secret',
        'endpoint_url': 'https://s3.amazonaws.com'
    }
}

storage = create_storage_manager(s3_config)

# List files
files = storage.list_files("s3://")

# Download file
storage.download_to_local("s3://my-file.hdr", "local-file.hdr")

# Upload file
storage.upload_from_local("local-file.nc", "s3://my-file.nc")
```

## 🎯 Example Programs

The library includes several example programs in the `examples/` folder:

### 1. `simple_example.py` - Basic Example
Basic processing of local CRISM files with minimal configuration.

```bash
cd examples
python simple_example.py
```

**Features:**
- Processes local SR and IF files
- Uses Dask for parallel processing
- Saves result in NetCDF format
- Error handling and logging

### 2. `simple_s3.py` - Single S3 Processing
Processing of a single CRISM file from S3/MinIO storage.

```bash
cd examples
python simple_s3.py
```

**Features:**
- Connection to MinIO S3-compatible storage
- Automatic download from S3
- Local processing
- Upload result to S3
- Automatic cleanup of temporary files

**Configuration:**
```python
# Target file to process
target_file = "frt00003e12_07_de166l_mtr1.hdr"

# S3 credentials (already configured)
s3_config = {
    'bucket_name': 'vamorini-test',
    'endpoint_url': 'https://minio-api.eagleprojects.cloud'
}
```

### 3. `simple_s3_all.py` - S3 Batch Processing
Batch processing of all CRISM files present in an S3 bucket.

```bash
cd examples
python simple_s3_all.py
```

**Features:**
- Automatic bucket scanning
- Parallel processing (up to 4 threads)
- Real-time progress tracking
- Robust error handling
- Detailed statistics
- Batch upload of results

**Output:**
```
📁 Files processed: 15/20
📊 Success rate: 75.0%
⏱️  Total time: 245.67 seconds
💾 Results saved in: output_s3_batch
☁️  Files uploaded to S3: 15
🎯 Average time per file: 16.38 seconds
```

### 4. Other Available Examples

- `cloud_example.py` - Cloud processing example
- `kubernetes_example.py` - Kubernetes deployment
- `spectral_dask.py` - Spectral processing with Dask
- `xarray_plot.py` - Visualization with xarray
- `export_ascii_wcs.py` - ASCII format export

## 🔬 Main Functions

### CRISM Processing

#### `process_crism_file()`
Processes CRISM files (SR + IF) and saves in NetCDF format with georeferencing.

```python
from astrogea.core import process_crism_file

ds = process_crism_file(
    base_sr_path="path/to/sr_file",      # SR file (without extension)
    base_if_path="path/to/if_file",      # IF file (without extension)
    output_nc_path="output.nc",          # Output file
    use_dask=True                        # Use Dask for parallel processing
)
```

#### `envi_to_xarray_wcs()`
Converts ENVI file to xarray with WCS coordinates.

```python
from astrogea.core import envi_to_xarray_wcs

ds = envi_to_xarray_wcs("file.hdr", "file.img")  # img optional
```

### Spectral Analysis

#### `continuum_removal()`
Continuum removal with convex hull.

```python
from astrogea.core import continuum_removal

result, wavelengths = continuum_removal(
    img,           # Hyperspectral cube (Y, X, bands)
    wavelength,    # Wavelength array
    MIN, MAX,      # Analysis range
    use_dask=True  # Parallel processing
)
```

#### `band_parameters_mafic()`
Mafic band parameter analysis.

```python
from astrogea.core import band_parameters_mafic

mafic_map = band_parameters_mafic(
    img_removed,   # Data after continuum removal
    wavelength,    # Wavelengths
    nbands=5,      # Number of bands to analyze
    use_dask=True  # Parallel processing
)
```

#### `smoothing_moving_average()`
Moving average smoothing.

```python
from astrogea.core import smoothing_moving_average

result, wavelengths = smoothing_moving_average(
    img,           # Hyperspectral cube
    wavelength,    # Wavelengths
    MIN, MAX,      # Analysis range
    window_size=5, # Window size
    use_dask=True  # Parallel processing
)
```

#### `coregister_spectra()`
Spectral coregistration.

```python
from astrogea.core import coregister_spectra

interpolated = coregister_spectra(
    reference_wavelengths,  # Reference grid
    new_wavelengths,        # Original wavelengths
    new_reflectance         # Reflectance values
)
```

### Storage and Configuration

#### `create_storage_manager()`
Creates manager for multi-cloud storage.

```python
from astrogea.storage import create_storage_manager

# S3 Configuration
config = {
    's3': {
        'bucket_name': 'my-bucket',
        'aws_access_key_id': 'key',
        'aws_secret_access_key': 'secret',
        'endpoint_url': 'https://s3.amazonaws.com'
    }
}

storage = create_storage_manager(config)
```

#### `create_config()`
Creates configuration for different environments.

```python
from astrogea.config import create_config, Environment

config = create_config(Environment.LOCAL)      # Local environment
config = create_config(Environment.KUBERNETES) # Kubernetes
config = create_config(Environment.DOCKER)     # Docker
```

### Spectral Utilities

```python
from astrogea.core import (
    remove_crism_bad_ranges_cube,  # Remove CRISM problematic ranges
    row_norm, column_norm,         # Row/column normalization
    center_norm, L1_norm,          # Centered/L1 normalization
    minmax, robust_scaler,         # Minmax/robust scaling
    derivative,                    # Spectral derivative
    log_1_r_norm,                  # Log(1/R) normalization
    baseline_correction_cube,      # Baseline correction
    auto_stretch_rgb,              # Automatic RGB stretch
    merge_datacubes,               # Datacube merge
    spetial_merge_datacubes,       # Spatial merge
    hypermerge_spatial             # Hyperspectral merge
)
```

## 🌐 Multi-Cloud Storage

AstroGea supports different cloud storage providers:

### Amazon S3 / MinIO
```python
s3_config = {
    's3': {
        'bucket_name': 'my-bucket',
        'aws_access_key_id': 'your-key',
        'aws_secret_access_key': 'your-secret',
        'region': 'us-east-1',
        'endpoint_url': 'https://s3.amazonaws.com'  # Opzionale per MinIO
    }
}
```

### Google Cloud Storage
```python
gcs_config = {
    'gcs': {
        'bucket_name': 'my-bucket',
        'project_id': 'my-project',
        'credentials_path': '/path/to/credentials.json'
    }
}
```

### Azure Blob Storage
```python
azure_config = {
    'azure': {
        'account_name': 'myaccount',
        'account_key': 'your-key',
        'container_name': 'my-container'
    }
}
```

## ⚡ Distributed Processing

### Dask Integration
All main functions support Dask for parallel processing:

```python
import dask.array as da
from astrogea.core import continuum_removal

# Create Dask array
img_dask = da.from_array(img, chunks=(100, 100, -1))

# Parallel processing
result, wavelengths = continuum_removal(
    img_dask, wavelength, MIN, MAX, use_dask=True
)
```

### Kubernetes Deployment
```bash
# Deploy to Kubernetes
kubectl apply -f k8s/
```

## 📊 Export and Visualization

### Multi-Format Export
```python
from astrogea.export import export_to_ascii_wcs, export_to_csv_wcs, export_to_geotiff_wcs

# ASCII export with WCS
export_to_ascii_wcs(dataset, "output.asc", variable="spectral_data")

# CSV export
export_to_csv_wcs(dataset, "output.csv", include_metadata=True)

# GeoTIFF export
export_to_geotiff_wcs(dataset, "output.tif", wavelength_idx=0)
```

### Visualization
```python
import matplotlib.pyplot as plt
import xarray as xr

# Load dataset
ds = xr.open_dataset("output.nc")

# Spectral plot
ds.spectral_data.isel(line=10, sample=10).plot()
plt.show()

# RGB plot
rgb = ds.spectral_data.isel(wavelength=[10, 20, 30])
rgb.plot.imshow(col='wavelength', size=4)
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Tests with coverage
pytest --cov=astrogea

# Specific tests
pytest tests/test_core.py
pytest tests/test_spectral.py
```

## 📈 Performance

### Processing Benchmarks
- **Typical CRISM file (640x640x438)**: ~30 seconds (with Dask)
- **Batch processing (20 files)**: ~8 minutes (4 parallel threads)
- **Memory usage**: ~2-4 GB per typical file
- **Storage**: ~50-100 MB per NetCDF output file

### Optimizations
- Use `use_dask=True` for large files
- Configure appropriate chunk size for Dask
- Use local storage for temporary files
- Enable NetCDF compression to reduce space

## 🔧 Advanced Configuration

### Environment Variables
```bash
export ASTROGEA_S3_BUCKET="my-bucket"
export ASTROGEA_S3_REGION="us-east-1"
export DASK_SCHEDULER_URL="tcp://localhost:8786"
export ASTROGEA_WORKERS="4"
```

### Configuration File
```yaml
# astrogea_config.yaml
environment: local
storage:
  s3:
    bucket_name: my-bucket
    region: us-east-1
  local:
    base_path: /data/local
processing:
  use_dask: true
  parallel_workers: 4
  memory_limit: 4GB
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is distributed under the MIT license. See `LICENSE` for more details.

## 👥 Authors

- **Valter Amorini** - *Initial development* - [valter.amorini@i4m.it](mailto:valter.amorini@i4m.it)

## 🙏 Acknowledgments

- NASA for CRISM data
- Scientific community for feedback and contributions
- Open source library developers

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/tuutente/astrogea/issues)
- **Email**: [valter.amorini@i4m.it](mailto:valter.amorini@i4m.it)
- **Documentation**: [GitHub Wiki](https://github.com/tuutente/astrogea/wiki)

---

**AstroGea** - Advanced CRISM spectral data processing for planetary science 🚀🔬