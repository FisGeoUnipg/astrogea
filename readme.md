# AstroGea 🌌

**AstroGea** is a Python library for processing CRISM (Compact Reconnaissance Imaging Spectrometer for Mars) spectral data in ENVI format, integrating spatial coordinates (WCS) and saving them in NetCDF format.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/astrogea.svg)](https://badge.fury.io/py/astrogea)

## 🚀 Features

- **CRISM Data Processing**: Load and process ENVI spectral data
- **Georeferencing**: Automatic integration of WCS spatial coordinates
- **NetCDF Format**: Save in standard scientific data format
- **CLI Integration**: Command-line interface for batch processing
- **xarray Support**: Native integration with Python scientific ecosystem
- **WCS Support**: Full World Coordinate System implementation using astropy
- **Performance Optimization**: Optional Dask support for large datasets

## 📦 Installation

### Installation from PyPI (recommended)

```bash
pip install astrogea[netcdf4]
```

### Installation from TestPyPI

```bash
pip install -i https://test.pypi.org/simple/ astrogea[netcdf4]
```

### Installation from source

```bash
git clone https://github.com/yourusername/astrogea.git
cd astrogea
pip install -e .[netcdf4]
```

## 🔧 Dependencies

### Core Dependencies
- **numpy** >= 1.20.0
- **xarray** >= 2022.1.0
- **spectral** >= 0.22.0
- **astropy** >= 5.0.0

### Optional Dependencies
- **netCDF4** >= 1.6.0 (for NetCDF compression)
- **dask** >= 2022.1.0 (for large dataset processing)
- **rasterio** >= 1.2.0 (for geospatial operations)

## 📖 Usage

### Command Line Interface

```bash
astrogea --sr /path/to/surface_reflectance --ifile /path/to/incidence_factor --out output.nc
```

Available options:
- `--sr`: Base path to Surface Reflectance .hdr file (without extension)
- `--ifile`: Base path to Incidence Factor .hdr file (without extension)
- `--out`: Output NetCDF file
- `--dask`: Use Dask for file loading (optional)

### Python Library Usage

#### Basic CRISM Processing

```python
import astrogea
from astrogea import process_crism_file

# Process CRISM files
dataset = process_crism_file(
    base_sr_path="/path/to/surface_reflectance",
    base_if_path="/path/to/incidence_factor", 
    output_nc_path="output.nc"
)

print(dataset)
```

#### Single ENVI File Processing

```python
from astrogea import envi_to_xarray_wcs

# Convert single ENVI file to xarray with WCS
ds = envi_to_xarray_wcs("data/file.hdr", "data/file.img")
print(ds)
```

#### Complete Example

```python
import astrogea
import xarray as xr

# Load and process data
ds = astrogea.process_crism_file(
    base_sr_path="data/FRT0000B6F7_07_IF166L_TRR3_SYS001",
    base_if_path="data/FRT0000B6F7_07_IF166L_TRR3_SYS001_IF",
    output_nc_path="processed_data.nc"
)

# Access data
surface_reflectance = ds.spectral_data
geometry_angles = ds.geometry_angles

# View coordinates
print(f"Dimensions: {ds.dims}")
print(f"Coordinates: {ds.coords}")
print(f"WCS available: {ds.attrs.get('has_wcs', 0)}")
```

## 🧪 Testing

Run unit tests:

```bash
python test_astrogea.py
```

Or with pytest:

```bash
pytest tests/
```

## 📚 API Documentation

### Core Functions

#### `process_crism_file(base_sr_path, base_if_path, output_nc_path, use_dask=False)`

Main function for processing CRISM files. Combines Surface Reflectance (SR) and Incidence Factor (IF) data into a single NetCDF dataset with WCS coordinates.

**Parameters:**
- `base_sr_path` (str): Base path to Surface Reflectance .hdr file
- `base_if_path` (str): Base path to Incidence Factor .hdr file
- `output_nc_path` (str): Output NetCDF file path
- `use_dask` (bool, optional): Use Dask for loading (default: False)

**Returns:**
- `xarray.Dataset`: Dataset with georeferenced data

**Features:**
- Automatic wavelength extraction from metadata
- WCS coordinate system creation from ENVI map info
- Global metadata preservation
- Optional Dask backend for large datasets

#### `envi_to_xarray_wcs(hdr_path, img_path=None)`

Convert a single ENVI file to xarray with WCS coordinates.

**Parameters:**
- `hdr_path` (str): Path to .hdr file
- `img_path` (str, optional): Path to .img file (if None, uses same name as hdr_path)

**Returns:**
- `xarray.Dataset`: Dataset with WCS coordinates

### WCS Utilities

#### `parse_envi_map_info_list(map_info_list)`

Parse ENVI map info list to extract georeferencing information.

**Parameters:**
- `map_info_list` (List[str]): ENVI map info list

**Returns:**
- `dict`: Parsed map information with projection, coordinates, and units

#### `create_wcs_from_parsed_info(parsed_info, shape)`

Create astropy WCS object from parsed map info.

**Parameters:**
- `parsed_info` (dict): Parsed map information
- `shape` (tuple): Data shape (lines, samples)

**Returns:**
- `astropy.wcs.WCS`: WCS object for coordinate transformations

#### `create_wcs_header_dict(wcs_object, parsed_info)`

Create WCS header dictionary for NetCDF attributes.

**Parameters:**
- `wcs_object` (astropy.wcs.WCS): WCS object
- `parsed_info` (dict): Parsed map information

**Returns:**
- `dict`: WCS header dictionary for NetCDF storage

### Spectral Processing

#### `SpectralArrayWrapper(data, wavelengths, wavelength_dim='wavelength')`

Wrapper for spectral arrays with wavelength coordinates.

**Methods:**
- `to_xarray()`: Convert to xarray.DataArray with proper dimensions

**Features:**
- Automatic dimension order detection
- Support for different CRISM data formats
- Wavelength coordinate integration

## 🔧 Performance Considerations

### NumPy vs Dask

The library supports both NumPy and Dask backends:

- **NumPy (default)**: Faster for medium-sized datasets (< 1GB)
- **Dask**: Better for very large datasets or distributed processing

**Performance test results:**
- NumPy: ~2.5s for 800MB dataset
- Dask: ~10.8s for same dataset (overhead for small files)

**Recommendation:** Use NumPy for typical CRISM files, Dask only for very large datasets.

### Memory Usage

- Typical memory usage: ~1x data size
- Efficient memory management with optional chunking
- Automatic cleanup of temporary objects

## 🌍 Supported Data Formats

### Input Formats
- **ENVI (.hdr/.img)**: Primary format for CRISM data
- **Spectral Python**: Native support via spectral library

### Output Formats
- **NetCDF (.nc)**: Primary output format with compression
- **CSV**: Textual output for analysis
- **FITS**: Astronomical format (if astropy available)

### Coordinate Systems
- **WCS**: World Coordinate System via astropy
- **UTM**: Universal Transverse Mercator
- **Geographic**: Latitude/Longitude coordinates
- **Custom projections**: Via ENVI map info

## 🚀 Advanced Usage

### Custom WCS Processing

```python
from astrogea import parse_envi_map_info_list, create_wcs_from_parsed_info

# Parse custom map info
map_info = ["UTM", "1.0", "1.0", "100.0", "200.0", "30.0", "30.0", "meters"]
parsed = parse_envi_map_info_list(map_info)

# Create WCS
wcs = create_wcs_from_parsed_info(parsed, (100, 200))
```

### Batch Processing

```python
import glob
from astrogea import envi_to_xarray_wcs

# Process multiple files
for hdr_file in glob.glob("data/*.hdr"):
    ds = envi_to_xarray_wcs(hdr_file)
    output_file = hdr_file.replace('.hdr', '_processed.nc')
    ds.to_netcdf(output_file)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 🙏 Acknowledgments

- NASA for CRISM data
- Python scientific community for base tools
- All contributors who made this project possible

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Contact: your@email.com

---

**AstroGea** - Martian spectral data processing 🌌
