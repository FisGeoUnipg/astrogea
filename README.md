# AstroGea

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/badge/pypi-astrogea-blue.svg)](https://pypi.org/project/astrogea/)

**AstroGea** is an advanced Python library for processing and analyzing CRISM (Compact Reconnaissance Imaging Spectrometer for Mars) spectral data with full support for WCS georeferencing and NetCDF formats. It includes tools for mafic band analysis, distributed processing, and cloud storage.

## Key Features

- **CRISM Processing**: Conversion of ENVI files (.hdr/.img) to georeferenced NetCDF datasets
- **Spectral Analysis**: Continuum removal, smoothing, spectral coregistration
- **Advanced Spectral Processing**: Row-wise/column-wise normalization, wavelength range extraction, advanced continuum removal
- **Machine Learning**: Autoencoder models for spectral dimensionality reduction and feature extraction
- **Mafic Analysis**: Extraction of mafic band parameters (minimum, center, depth, area, asymmetry)
- **Multi-Cloud Storage**: Support for S3, Google Cloud Storage, Azure Blob Storage
- **Distributed Processing**: Dask support for parallel processing
- **Georeferencing**: Complete WCS integration for geographic coordinates
- **Multi-Format Export**: NetCDF, CSV, GeoTIFF, ASCII
- **Visualization**: Comprehensive plotting and visualization tools

## Installation

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
git clone https://github.com/FisGeoUnipg/astrogea.git
cd astrogea
pip install -e .
```

## Dependencies

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
- `torch>=1.10.0` - Machine learning functions (autoencoder)

## Quick Start Guide

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

## Example Programs

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
Files processed: 15/20
Success rate: 75.0%
Total time: 245.67 seconds
Results saved in: output_s3_batch
Files uploaded to S3: 15
Average time per file: 16.38 seconds
```

### 4. `spectral_example.py` - Advanced Spectral Processing
Comprehensive example demonstrating advanced spectral processing functions including normalization, continuum removal, dimension reduction, and machine learning capabilities.

```bash
cd examples
python spectral_example.py
```

**Features:**
- Direct loading from `.img` files
- Advanced normalization methods (row-wise integral, column-wise)
- Continuum removal with convex hull method
- Wavelength range-based dimension reduction
- Machine learning autoencoder for dimensionality reduction
- Comprehensive visualization plots

**Output:**
- Processed spectra with various normalizations
- Continuum-removed spectra
- Reduced-dimension spectra
- Visualization plots saved in `output_plots/` directory
- Statistics and analysis results

### 5. Other Available Examples

- `cloud_example.py` - Cloud processing example
- `kubernetes_example.py` - Kubernetes deployment
- `spectral_dask.py` - Spectral processing with Dask
- `xarray_plot.py` - Visualization with xarray
- `export_ascii_wcs.py` - ASCII format export

## Main Functions

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

## Advanced Spectral Processing Functions

AstroGea includes advanced spectral processing functions for comprehensive data analysis, normalization, continuum removal, and machine learning-based dimensionality reduction.

### Advanced Normalization

#### `row_wise_integral_norm_data()`
Normalizes each spectrum by its integral over wavelength (row-wise integral normalization). This method preserves the spectral shape while normalizing the total energy.

```python
from astrogea.core import row_wise_integral_norm_data
import numpy as np

# Example: Normalize spectra by their integral
wavelength = np.linspace(1.0, 2.5, 60)
spectra = np.random.rand(1000, 60) * 1000  # (n_spectra, n_bands)

normalized_spectra, wav = row_wise_integral_norm_data(
    wavelength,
    spectra,
    use_dask=True  # Enable parallel processing
)

print(f"Normalized spectra shape: {normalized_spectra.shape}")
```

#### `column_wise_norm_advanced()`
Advanced column-wise normalization with Dask support. Normalizes each wavelength band to zero mean and unit variance across all spectra.

```python
from astrogea.core import column_wise_norm_advanced

# Normalize each wavelength band
normalized = column_wise_norm_advanced(
    spectra,
    use_dask=True  # Parallel processing
)

# Verify normalization: mean ≈ 0, std ≈ 1
print(f"Mean per column: {np.mean(normalized, axis=0)[:5]}")
print(f"Std per column: {np.std(normalized, axis=0)[:5]}")
```

### Wavelength Utilities

#### `find_nearest_wavelength()`
Finds the index of the nearest wavelength value in an array. Useful for selecting specific spectral bands.

```python
from astrogea.core import find_nearest_wavelength

wavelength = np.linspace(1.0, 2.5, 60)
target = 1.5  # um

idx = find_nearest_wavelength(wavelength, target)
print(f"Nearest wavelength to {target} um: {wavelength[idx]:.4f} um at index {idx}")
```

### Continuum Removal

#### `continuum_removal_points()`
Performs continuum removal on a single spectrum using the convex hull method. This is useful for isolating absorption features.

```python
from astrogea.core import continuum_removal_points
import numpy as np

# Create spectrum points (wavelength, reflectance)
wavelength = np.linspace(1.0, 2.5, 60)
reflectance = np.random.rand(60) * 0.5 + 0.5
points = np.column_stack([wavelength, reflectance])

# Remove continuum
removed = continuum_removal_points(points, interp_type='linear')

# Plot comparison
import matplotlib.pyplot as plt
plt.plot(wavelength, reflectance, 'b-', label='Original')
plt.plot(wavelength, removed, 'r-', label='Continuum Removed')
plt.legend()
plt.show()
```

#### `compute_removal_single()`
Computes continuum removal for a single spectrum given wavelength and reflectance arrays.

```python
from astrogea.core import compute_removal_single

wavelength = np.linspace(1.0, 2.5, 60)
spectrum = np.random.rand(60) * 0.5 + 0.5

removed = compute_removal_single(
    wavelength,
    spectrum,
    interp_type='linear'  # or 'cubic', 'quadratic', etc.
)
```

#### `compute_removal()`
Computes continuum removal for multiple spectra in batch.

```python
from astrogea.core import compute_removal

# Multiple spectra
spectra = np.random.rand(100, 60) * 0.5 + 0.5  # (n_spectra, n_bands)
wavelength = np.linspace(1.0, 2.5, 60)

# Remove continuum for all spectra
removed_spectra = compute_removal(
    wavelength,
    spectra,
    interp_type='linear'
)

print(f"Removed spectra shape: {removed_spectra.shape}")
```

### Dimension Reduction

#### `dimension_reduction()`
Extracts spectra from an image within a specified wavelength range, optionally applying continuum removal.

```python
from astrogea.core import dimension_reduction
import numpy as np

# Load hyperspectral image (height, width, bands)
img = np.random.rand(588, 729, 60) * 1000
wavelength = np.linspace(1.0, 2.5, 60)

# Extract spectra in wavelength range 1.2 - 1.8 um
spectra, indexes, reduced_wavelength = dimension_reduction(
    img,
    w1=1.2,           # Start wavelength (um)
    w2=1.8,           # End wavelength (um)
    wavelength=wavelength,
    cr=False          # Set to True to apply continuum removal
)

print(f"Extracted {spectra.shape[0]} spectra")
print(f"Wavelength range: {reduced_wavelength[0]:.4f} - {reduced_wavelength[-1]:.4f} um")
print(f"Number of bands: {len(reduced_wavelength)}")
```

#### `dimension_reduction_spectral_parameters()`
Extracts spectra from an image with optional normalization chain. This is the most comprehensive extraction function.

```python
from astrogea.core import dimension_reduction_spectral_parameters

# Extract spectra with normalization
spectra, indexes = dimension_reduction_spectral_parameters(
    img,                    # 3D image array (height, width, bands)
    norm=['column'],        # Normalization methods: 'row', 'column', 'minmax', 'L1'
    zeros=True,             # Set negative values to zero
    use_dask=False          # Enable Dask for parallel processing
)

print(f"Extracted {spectra.shape[0]} valid spectra")
print(f"Each spectrum has {spectra.shape[1]} bands")
print(f"Pixel coordinates shape: {indexes.shape}")

# Access pixel coordinates
for i in range(10):
    line, sample = indexes[i]
    print(f"Spectrum {i}: pixel at line={line}, sample={sample}")
```

**Normalization options:**
- `'row'`: Normalize each spectrum by its L1 norm
- `'column'`: Normalize each wavelength band to zero mean and unit variance
- `'minmax'`: Min-max normalization per wavelength band
- `'L1'`: L1 normalization per wavelength band
- `'none'` or `None`: No normalization

### Advanced Auto-Stretch

#### `auto_stretch_rgb_advanced()`
Advanced auto-stretch RGB function with linearization and normalization options. Can return either a stretched cube or linearized spectra.

```python
from astrogea.core import auto_stretch_rgb_advanced
import numpy as np

# Hyperspectral image
img_sr = np.random.rand(588, 729, 60) * 1000
product_names = [f'Band_{i}' for i in range(60)]

# Option 1: Return stretched cube
stretched_cube = auto_stretch_rgb_advanced(
    img_sr,
    product_names,
    n_bins=1000,
    plot=False,           # Set to True to show histograms
    linearize=False      # Return cube, not spectra
)

# Option 2: Return linearized spectra with normalization
spectra, indexes = auto_stretch_rgb_advanced(
    img_sr,
    product_names,
    n_bins=1000,
    plot=False,
    linearize=True,      # Return linearized spectra
    norm=['column'],     # Apply column normalization
    zeros=True           # Set negative values to zero
)

print(f"Linearized spectra shape: {spectra.shape}")
print(f"Pixel indexes shape: {indexes.shape}")
```

## Machine Learning Functions

AstroGea includes machine learning capabilities for spectral data analysis, particularly autoencoders for dimensionality reduction and feature extraction.

**Note:** ML functions require PyTorch. Install with: `pip install astrogea[ml]` or `pip install torch`

### Autoencoder Models

#### `SpectralAutoencoder`
Neural network autoencoder for spectral data dimensionality reduction.

```python
from astrogea.ml import SpectralAutoencoder, weight_init
import torch
import torch.nn as nn

# Prepare data
spectra = np.random.rand(1000, 60)  # (n_spectra, n_bands)
spectra_tensor = torch.FloatTensor(spectra)

# Create autoencoder
model = SpectralAutoencoder(
    encoded_space_dim=10,    # Dimension of encoded space
    in_channels=60,          # Number of input spectral bands
    n_layers_encoder=2,     # Number of encoder layers
    n_layers_decoder=2,      # Number of decoder layers
    out1=[128, 64],         # Encoder layer sizes
    out2=[64, 128],         # Decoder layer sizes
    act=nn.ReLU(),          # Activation function
    drops=[0.2, 0.2, 0.2],  # Dropout values
    last=True               # Unused parameter
)

# Initialize weights
weight_init(model, init_method='xavier_normal')

# Forward pass
with torch.no_grad():
    encoded = model.encoder(spectra_tensor[:10])
    reconstructed = model(spectra_tensor[:10])

print(f"Input shape: {spectra_tensor[:10].shape}")
print(f"Encoded shape: {encoded.shape}")
print(f"Reconstructed shape: {reconstructed.shape}")
```

#### `random_search_autoencoder()`
Random search for optimal autoencoder hyperparameters.

```python
from astrogea.ml import random_search_autoencoder
from torch.utils.data import TensorDataset
import torch.nn as nn

# Prepare dataset
dataset = TensorDataset(torch.FloatTensor(spectra))

# Search for best hyperparameters
best_lr, best_enc, best_dec, best_w, best_dim, best_act, best_drops, best_init, best_l1 = \
    random_search_autoencoder(
        dataset=dataset,
        in_channels=60,
        criterion=nn.MSELoss(),
        encoded_space_dim=[5, 10, 15, 20],  # Possible dimensions
        n_encmax=3,                         # Max encoder layers
        n_decmax=3,                         # Max decoder layers
        MINenc=32, MAXenc=256,              # Encoder layer size range
        MINdec=32, MAXdec=256,              # Decoder layer size range
        activations=[nn.ReLU(), nn.Tanh()],
        try_epochs=10,                      # Training epochs per trial
        N_try=20,                           # Number of random trials
        bs=100,                             # Batch size
        val_split=0.2                       # Validation split
    )

print(f"Best learning rate: {best_lr}")
print(f"Best encoded dimension: {best_dim}")
print(f"Best encoder layers: {best_enc}")
print(f"Best decoder layers: {best_dec}")
```

#### `train_autoencoder()`
Train an autoencoder model with validation and learning rate scheduling.

```python
from astrogea.ml import train_autoencoder
from torch.utils.data import TensorDataset, random_split
import torch.nn as nn

# Prepare datasets
dataset = TensorDataset(torch.FloatTensor(spectra))
train_len = int(len(dataset) * 0.8)
train_set, val_set = random_split(dataset, [train_len, len(dataset) - train_len])

# Create model
model = SpectralAutoencoder(
    encoded_space_dim=10,
    in_channels=60,
    n_layers_encoder=2,
    n_layers_decoder=2,
    out1=[128, 64],
    out2=[64, 128],
    act=nn.ReLU(),
    drops=[0.2, 0.2, 0.2],
    last=True
)

# Train model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_losses, val_losses = train_autoencoder(
    model=model,
    criterion=nn.MSELoss(),
    train_ds=train_set,
    validation_ds=val_set,
    num_epochs=50,
    patience=5,              # Learning rate scheduler patience
    weight_decay=1e-5,
    bs=1024,                 # Batch size (fixed to 1024)
    device=device,
    LR=0.001,
    printer=True             # Print training progress
)

# Plot training curves
import matplotlib.pyplot as plt
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

#### `plot_encoded_space()`
Visualize the encoded space of a trained autoencoder.

```python
from astrogea.ml import plot_encoded_space

# Visualize encoded space (2D projections)
encoded_data = plot_encoded_space(
    spectra[:1000],          # Sample spectra
    model,                   # Trained model
    DIM=5,                   # Number of dimensions to plot
    plot=True                # Display plot
)

print(f"Encoded data shape: {encoded_data.shape}")
```

### Complete ML Workflow Example

```python
import numpy as np
import torch
import torch.nn as nn
from astrogea.core import dimension_reduction_spectral_parameters
from astrogea.ml import SpectralAutoencoder, train_autoencoder, plot_encoded_space
from torch.utils.data import TensorDataset, random_split

# 1. Extract spectra from image
spectra, indexes = dimension_reduction_spectral_parameters(
    img_array,
    norm=['column'],
    zeros=True,
    use_dask=False
)

# 2. Prepare PyTorch dataset
dataset = TensorDataset(torch.FloatTensor(spectra))
train_set, val_set = random_split(dataset, [0.8, 0.2])

# 3. Create autoencoder
model = SpectralAutoencoder(
    encoded_space_dim=10,
    in_channels=spectra.shape[1],
    n_layers_encoder=2,
    n_layers_decoder=2,
    out1=[128, 64],
    out2=[64, 128],
    act=nn.ReLU(),
    drops=[0.2, 0.2, 0.2],
    last=True
)

# 4. Train model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_losses, val_losses = train_autoencoder(
    model, nn.MSELoss(), train_set, val_set,
    num_epochs=50, patience=5, weight_decay=1e-5,
    bs=1024, device=device, LR=0.001, printer=True
)

# 5. Visualize encoded space
encoded = plot_encoded_space(spectra[:1000], model, DIM=5, plot=True)

# 6. Use encoded features for analysis
with torch.no_grad():
    encoded_features = model.encoder(torch.FloatTensor(spectra))
    print(f"Encoded features shape: {encoded_features.shape}")
```

## Multi-Cloud Storage

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

## Distributed Processing

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

## Export and Visualization

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

## Testing

```bash
# Run all tests
pytest

# Tests with coverage
pytest --cov=astrogea

# Specific tests
pytest tests/test_core.py
pytest tests/test_spectral.py
```

## Performance

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

## Advanced Configuration

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

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

This project is distributed under the MIT license. See `LICENSE` for more details.

## Authors

- **Valter Amorini** - *Initial development* - [valter.amorini@i4m.it](mailto:valter.amorini@i4m.it)

## Acknowledgments

- NASA for CRISM data
- Scientific community for feedback and contributions
- Open source library developers

## Support

- **Issues**: [GitHub Issues](https://github.com/FisGeoUnipg/astrogea/issues)
- **Email**: [valter.amorini@i4m.it](mailto:valter.amorini@i4m.it)
- **Documentation**: [GitHub Wiki](https://github.com/FisGeoUnipg/astrogea/wiki)

---

**AstroGea** - Advanced CRISM spectral data processing for planetary science