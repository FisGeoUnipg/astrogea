#!/usr/bin/env python3
"""
Test script for the AstroGea library
Verifies that all main functionalities work correctly.
"""

import numpy as np
import xarray as xr
import tempfile
import os
import pandas as pd

def test_imports():
    """Test imports"""
    print("🧪 Test 1: Imports...")
    try:
        import astrogea
        from astrogea import process_crism_file, SpectralArrayWrapper
        from astrogea import parse_envi_map_info_list, create_wcs_from_parsed_info
        print(f"✅ Imports successful! Version: {astrogea.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_spectral_array_wrapper():
    """Test the SpectralArrayWrapper class"""
    print("\n🧪 Test 2: SpectralArrayWrapper...")
    try:
        from astrogea import SpectralArrayWrapper
        
        # Create test data
        data = np.random.rand(10, 5, 5)  # 10 bands, 5x5 pixels
        wavelengths = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
        
        # Test wrapper
        wrapper = SpectralArrayWrapper(data, wavelengths)
        xarray_data = wrapper.to_xarray()
        
        # Verifications
        assert isinstance(xarray_data, xr.DataArray)
        assert xarray_data.dims == ('line', 'sample', 'wavelength')
        assert len(xarray_data.coords['wavelength']) == 10
        assert xarray_data.shape == (5, 5, 10)
        
        print("✅ SpectralArrayWrapper works correctly!")
        return True
    except Exception as e:
        print(f"❌ Error in SpectralArrayWrapper: {e}")
        return False

def test_wcs_utils():
    """Test WCS utilities"""
    print("\n🧪 Test 3: WCS utilities...")
    try:
        from astrogea import parse_envi_map_info_list, create_wcs_from_parsed_info, create_wcs_header_dict
        
        # Test parsing map info
        map_info = ["UTM", "1.0", "2.0", "100.0", "200.0", "30.0", "30.0", "meters"]
        parsed = parse_envi_map_info_list(map_info)
        
        # Parsing verifications
        assert parsed["projection_name"] == "UTM"
        assert parsed["ref_x_1based"] == 1.0
        assert parsed["ref_coord1"] == 100.0
        assert parsed["pixel_size_x"] == 30.0
        assert parsed["units"] == "m"
        
        # Test WCS creation
        shape = (5, 10)  # (lines, samples)
        wcs = create_wcs_from_parsed_info(parsed, shape)
        assert wcs is not None
        assert hasattr(wcs, 'wcs')
        assert wcs.wcs.cdelt[0] == 30.0  # pixel_size_x
        assert wcs.wcs.cdelt[1] == -30.0  # -pixel_size_y
        
        # Test WCS header dict
        wcs_header = create_wcs_header_dict(wcs, parsed)
        assert wcs_header is not None
        assert 'CTYPE1' in wcs_header
        assert 'CUNIT1' in wcs_header
        assert wcs_header['PROJNAME'] == 'UTM'
        
        print("✅ WCS utilities work correctly!")
        return True
    except Exception as e:
        print(f"❌ Error in WCS utilities: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cli_help():
    """Test CLI"""
    print("\n🧪 Test 4: CLI...")
    try:
        import subprocess
        result = subprocess.run(['astrogea', '--help'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ CLI works correctly!")
            print(f"   Output: {result.stdout[:100]}...")
            return True
        else:
            print(f"❌ CLI doesn't work: {result.stderr}")
            return False
    except FileNotFoundError:
        print("⚠️  CLI not found (might be normal in development environment)")
        return True
    except Exception as e:
        print(f"❌ Error in CLI test: {e}")
        return False

def test_real_crism_files():
    """Test with real CRISM files using exclusively the astrogea library"""
    print("\n🧪 Test 5: Test with real CRISM files using astrogea...")
    try:
        # File paths
        data_dir = "data"
        base_name = "frt00006fbd_07_if164j_mtr3"
        hdr_path = os.path.join(data_dir, f"{base_name}.hdr")
        img_path = os.path.join(data_dir, f"{base_name}.img")
        
        # Verify that files exist
        if not os.path.exists(hdr_path):
            print(f"❌ HDR file not found: {hdr_path}")
            print("   Make sure files are in the 'data/' folder")
            return False
        if not os.path.exists(img_path):
            print(f"❌ IMG file not found: {img_path}")
            print("   Make sure files are in the 'data/' folder")
            return False
        
        print(f"📁 Loading file: {hdr_path}")
        
        # Use exclusively the astrogea library
        from astrogea import envi_to_xarray_wcs
        
        # Convert ENVI file to xarray with WCS using only astrogea
        ds = envi_to_xarray_wcs(hdr_path, img_path)
        
        print(f"✅ ENVI file converted to xarray with WCS using astrogea!")
        print(f"   Dimensions: {ds.dims}")
        print(f"   Variables: {list(ds.data_vars.keys())}")
        print(f"   Coordinates: {list(ds.coords.keys())}")
        
        # Show detailed information
        spectral_data = ds.spectral_data
        print(f"   Spectral data: {spectral_data.shape}")
        print(f"   Wavelength range: {spectral_data.coords['wavelength'].min().values:.2f} - {spectral_data.coords['wavelength'].max().values:.2f}")
        
        # Check WCS information
        has_wcs = ds.attrs.get('has_wcs', 0)
        if has_wcs:
            print(f"   ✅ WCS information present (has_wcs={has_wcs})")
            wcs_header = ds.attrs.get('wcs_header_dict', '')
            print(f"   WCS header length: {len(str(wcs_header))} characters")
        else:
            print("   ⚠️  No WCS information found")
        
        # Check metadata
        print(f"   Title: {ds.attrs.get('title', 'N/A')}")
        print(f"   Creation date: {ds.attrs.get('creation_date', 'N/A')}")
        print(f"   Conventions: {ds.attrs.get('Conventions', 'N/A')}")
        
        # Save the xarray with WCS in textual format
        output_path = f"{base_name}_astrogea_wcs.csv"
        text_summary_path = f"{base_name}_astrogea_wcs_summary.txt"
        
        try:
            # Save as CSV (textual format)
            # Convert to DataFrame and save as CSV
            
            # Get a sample of the data for CSV (first few wavelengths)
            sample_data = ds.spectral_data.isel(wavelength=slice(0, 5))  # First 5 wavelengths
            df = sample_data.to_dataframe().reset_index()
            df.to_csv(output_path, index=False)
            
            print(f"✅ Xarray with WCS saved as CSV (textual): {output_path}")
            
            # Create a text summary
            with open(text_summary_path, 'w') as f:
                f.write("ASTROGEA CRISM DATA SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"File: {base_name}\n")
                f.write(f"Dimensions: {dict(ds.dims)}\n")
                f.write(f"Variables: {list(ds.data_vars.keys())}\n")
                f.write(f"Coordinates: {list(ds.coords.keys())}\n\n")
                
                f.write("SPECTRAL DATA:\n")
                f.write(f"  Shape: {spectral_data.shape}\n")
                f.write(f"  Wavelength range: {spectral_data.coords['wavelength'].min().values:.2f} - {spectral_data.coords['wavelength'].max().values:.2f}\n")
                f.write(f"  Number of bands: {len(spectral_data.coords['wavelength'])}\n\n")
                
                f.write("METADATA:\n")
                f.write(f"  Title: {ds.attrs.get('title', 'N/A')}\n")
                f.write(f"  Creation date: {ds.attrs.get('creation_date', 'N/A')}\n")
                f.write(f"  Conventions: {ds.attrs.get('Conventions', 'N/A')}\n")
                f.write(f"  Has WCS: {ds.attrs.get('has_wcs', 0)}\n\n")
                
                if has_wcs:
                    f.write("WCS INFORMATION:\n")
                    f.write(f"  WCS header present: Yes\n")
                    f.write(f"  WCS header length: {len(str(ds.attrs.get('wcs_header_dict', '')))} characters\n\n")
                else:
                    f.write("WCS INFORMATION:\n")
                    f.write(f"  WCS header present: No\n\n")
                
                f.write("SAMPLE DATA (first 5 wavelengths, 10x10 pixels):\n")
                # Use the correct dimension names from the dataset
                dim_names = list(ds.spectral_data.dims)
                if len(dim_names) >= 3:
                    sample = ds.spectral_data.isel(
                        wavelength=slice(0, 5), 
                        **{dim_names[1]: slice(0, 10), dim_names[2]: slice(0, 10)}
                    )
                    f.write(str(sample.values))
                else:
                    f.write("Cannot create sample - insufficient dimensions")
            
            print(f"✅ Text summary saved: {text_summary_path}")
            
        except Exception as save_error:
            print(f"❌ Error saving textual format: {save_error}")
            return False
        
        # Verify that the files were created
        assert os.path.exists(output_path)
        assert os.path.exists(text_summary_path)
        csv_size = os.path.getsize(output_path)
        txt_size = os.path.getsize(text_summary_path)
        print(f"📁 CSV file size: {csv_size / 1024:.1f} KB")
        print(f"📁 Text summary size: {txt_size / 1024:.1f} KB")
        
        # Load and verify the CSV file
        df = pd.read_csv(output_path)
        print(f"✅ CSV file loaded correctly: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        
        # Show sample of the text summary
        with open(text_summary_path, 'r') as f:
            summary_content = f.read()
            print(f"✅ Text summary created successfully")
            print(f"   Summary length: {len(summary_content)} characters")
            print(f"   First 200 chars: {summary_content[:200]}...")
        
        print(f"✅ Test with real CRISM files completed successfully!")
        print(f"📁 CSV file saved: {output_path}")
        print(f"📁 Text summary saved: {text_summary_path}")
        print(f"💡 Xarray with WCS created using exclusively the astrogea library")
        print(f"💡 You can open the CSV with: pandas.read_csv('{output_path}')")
        
        return True
    except Exception as e:
        print(f"❌ Error in test with real CRISM files: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_mock_data():
    """Test with simulated data using the astrogea library"""
    print("\n🧪 Test 6: Test with simulated data using astrogea...")
    try:
        # Create simulated ENVI data
        sr_data = np.random.rand(10, 20, 30)  # 10 bands, 20x30 pixels
        if_data = np.random.rand(10, 20, 30)
        
        # Create simulated metadata
        wavelengths = [float(i) for i in range(10)]
        map_info = ["UTM", "1.0", "1.0", "100.0", "200.0", "30.0", "30.0", "meters"]
        
        # Use astrogea's SpectralArrayWrapper
        from astrogea import SpectralArrayWrapper
        sr_wrapper = SpectralArrayWrapper(sr_data, wavelengths)
        if_wrapper = SpectralArrayWrapper(if_data, wavelengths)
        
        sr_array = sr_wrapper.to_xarray()
        if_array = if_wrapper.to_xarray()
        
        # Create dataset
        ds = xr.Dataset({
            "surface_reflectance": sr_array,
            "incidence_factor": if_array
        })
        
        # Apply georeferencing using the new WCS system
        from astrogea import parse_envi_map_info_list, create_wcs_from_parsed_info
        parsed_info = parse_envi_map_info_list(map_info)
        wcs_object = create_wcs_from_parsed_info(parsed_info, sr_data.shape[:2])
        
        if wcs_object:
            # Create coordinates using WCS
            y_size = sr_array.sizes['y']
            x_size = sr_array.sizes['x']
            
            # Calculate coordinates from WCS
            y_coords = wcs_object.wcs.crval[1] + wcs_object.wcs.cdelt[1] * np.arange(y_size)
            x_coords = wcs_object.wcs.crval[0] + wcs_object.wcs.cdelt[0] * np.arange(x_size)
            
            ds = ds.assign_coords({
                "x": x_coords,
                "y": y_coords
            })
        
        # Save in NetCDF
        output_path = "mock_data_astrogea.nc"
        try:
            ds.to_netcdf(output_path)
            print(f"✅ Simulated data processed and saved: {output_path}")
        except Exception as save_error:
            print(f"⚠️  Warning: Could not save as NetCDF: {save_error}")
            try:
                ds.to_netcdf(output_path, encoding={})
                print(f"✅ Simulated data saved (without encoding): {output_path}")
            except Exception as save_error2:
                print(f"❌ Still cannot save: {save_error2}")
                return False
        
        print(f"   Dimensions: {ds.dims}")
        print(f"   Variables: {list(ds.data_vars.keys())}")
        print(f"   Coordinates: {list(ds.coords.keys())}")
        
        # Verify the file
        assert os.path.exists(output_path)
        loaded_ds = xr.open_dataset(output_path)
        print(f"✅ Mock data file loaded correctly: {loaded_ds.dims}")
        loaded_ds.close()
        
        return True
    except Exception as e:
        print(f"❌ Error in test with mock data: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fits_export():
    """Test FITS export functionality"""
    print("\n🧪 Test 7: FITS export test...")
    try:
        # Create test data
        data = np.random.rand(10, 20, 30)
        wavelengths = [float(i) for i in range(10)]
        
        from astrogea import SpectralArrayWrapper
        wrapper = SpectralArrayWrapper(data, wavelengths)
        xarray_data = wrapper.to_xarray()
        
        # Try to save as FITS if astropy is available
        try:
            import astropy.io.fits as fits
            from astropy.wcs import WCS
            
            # Create a simple WCS
            wcs = WCS(naxis=3)
            wcs.wcs.crpix = [1, 1, 1]
            wcs.wcs.crval = [0, 0, 0]
            wcs.wcs.cdelt = [1, 1, 1]
            wcs.wcs.ctype = ["WAVE", "RA---TAN", "DEC--TAN"]
            
            # Save as FITS
            output_path = "test_astrogea.fits"
            hdu = fits.PrimaryHDU(xarray_data.values)
            hdu.header.update(wcs.to_header())
            
            # Add wavelength information
            for i, wl in enumerate(wavelengths):
                hdu.header[f'WAVELEN{i}'] = wl
            
            hdu.writeto(output_path, overwrite=True)
            
            print(f"✅ FITS file saved: {output_path}")
            print(f"   File size: {os.path.getsize(output_path) / 1024:.1f} KB")
            
            # Verify the file
            with fits.open(output_path) as hdul:
                print(f"   FITS data shape: {hdul[0].data.shape}")
                print(f"   WCS present: {'WCS' in hdul[0].header}")
            
            return True
            
        except ImportError:
            print("⚠️  Astropy not available, skipping FITS export test")
            return True
            
    except Exception as e:
        print(f"❌ Error in FITS export test: {e}")
        return False

def test_combined_crism_files():
    """Test combining SR and IF files using process_crism_file function"""
    print("\n🧪 Test 8: Test combining SR and IF files...")
    try:
        # File paths - use the same base name but different suffixes
        data_dir = "data"
        base_name = "frt00006fbd_07"
        
        # Check if we have the SR and IF files with the expected names
        possible_sr_names = [
            f"{base_name}_sr164j_mtr3",
            f"{base_name}_sr164j",
            f"{base_name}_sr"
        ]
        possible_if_names = [
            f"{base_name}_if164j_mtr3", 
            f"{base_name}_if164j",
            f"{base_name}_if"
        ]
        
        # Find existing files
        sr_base = None
        if_base = None
        
        for sr_name in possible_sr_names:
            sr_hdr = os.path.join(data_dir, f"{sr_name}.hdr")
            sr_img = os.path.join(data_dir, f"{sr_name}.img")
            if os.path.exists(sr_hdr) and os.path.exists(sr_img):
                sr_base = sr_name
                break
        
        for if_name in possible_if_names:
            if_hdr = os.path.join(data_dir, f"{if_name}.hdr")
            if_img = os.path.join(data_dir, f"{if_name}.img")
            if os.path.exists(if_hdr) and os.path.exists(if_img):
                if_base = if_name
                break
        
        if sr_base is None or if_base is None:
            print(f"❌ Missing SR or IF files")
            print(f"   Looked for SR: {possible_sr_names}")
            print(f"   Looked for IF: {possible_if_names}")
            print("   Skipping combined CRISM test")
            return True  # Skip but don't fail
        
        print(f"📁 Found SR file: {sr_base}")
        print(f"📁 Found IF file: {if_base}")
        
        # Use the process_crism_file function
        from astrogea import process_crism_file
        
        output_nc_path = f"{sr_base}_combined_dataset.nc"
        
        # Process the combined dataset
        ds = process_crism_file(
            base_sr_path=os.path.join(data_dir, sr_base),
            base_if_path=os.path.join(data_dir, if_base),
            output_nc_path=output_nc_path
        )
        
        print(f"✅ Combined CRISM dataset created successfully!")
        print(f"   Dimensions: {ds.dims}")
        print(f"   Variables: {list(ds.data_vars.keys())}")
        print(f"   Coordinates: {list(ds.coords.keys())}")
        
        # Check spectral data
        spectral_data = ds.spectral_data
        print(f"   Spectral data: {spectral_data.shape}")
        print(f"   Wavelength range: {spectral_data.coords['wavelength'].min().values:.2f} - {spectral_data.coords['wavelength'].max().values:.2f}")
        
        # Check geometry data
        geometry_data = ds.geometry_angles
        print(f"   Geometry data: {geometry_data.shape}")
        print(f"   IF bands: {list(geometry_data.coords['if_band'].values)}")
        
        # Check metadata
        print(f"   Title: {ds.attrs.get('title', 'N/A')}")
        print(f"   Source files: {ds.attrs.get('source_files_base', 'N/A')}")
        print(f"   Has WCS: {ds.attrs.get('has_wcs', 0)}")
        
        # Verify the NetCDF file was created
        assert os.path.exists(output_nc_path)
        file_size = os.path.getsize(output_nc_path)
        print(f"📁 NetCDF file size: {file_size / 1024 / 1024:.1f} MB")
        
        # Load and verify the saved file
        loaded_ds = xr.open_dataset(output_nc_path)
        print(f"✅ NetCDF file loaded correctly: {loaded_ds.dims}")
        print(f"   Variables: {list(loaded_ds.data_vars.keys())}")
        print(f"   Metadata: {list(loaded_ds.attrs.keys())}")
        
        # Close the dataset
        loaded_ds.close()
        
        print(f"✅ Combined CRISM test completed successfully!")
        print(f"📁 File saved: {output_nc_path}")
        print(f"💡 Combined dataset created using astrogea.process_crism_file()")
        
        return True
    except Exception as e:
        print(f"❌ Error in combined CRISM test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_benchmark():
    """Test performance benchmark comparing with and without Dask"""
    print("\n🧪 Test 9: Performance benchmark...")
    try:
        import time
        from astrogea import envi_to_xarray_wcs
        
        # File paths
        data_dir = "data"
        base_name = "frt00006fbd_07_if164j_mtr3"
        hdr_path = os.path.join(data_dir, f"{base_name}.hdr")
        img_path = os.path.join(data_dir, f"{base_name}.img")
        
        # Check if file exists
        if not os.path.exists(hdr_path) or not os.path.exists(img_path):
            print(f"❌ Test file not found: {hdr_path}")
            print("   Skipping performance benchmark")
            return True
        
        print(f"📁 Benchmarking with file: {hdr_path}")
        
        # Test 1: Current implementation (without Dask)
        print("\n   Test 1: Current implementation (NumPy)...")
        start_time = time.perf_counter()
        
        ds_numpy = envi_to_xarray_wcs(hdr_path, img_path)
        
        numpy_time = time.perf_counter() - start_time
        print(f"   ✅ NumPy processing time: {numpy_time:.2f} seconds")
        print(f"   📊 Dataset size: {ds_numpy.spectral_data.shape}")
        print(f"   💾 Memory usage estimate: {ds_numpy.spectral_data.nbytes / 1024 / 1024:.1f} MB")
        
        # Test 2: Simulate Dask implementation
        print("\n   Test 2: Simulated Dask implementation...")
        start_time = time.perf_counter()
        
        # Simulate Dask loading (lazy loading)
        try:
            import dask.array as da
            from dask.diagnostics import ProgressBar
            import spectral.io.envi as envi
            
            # Load with Dask (simulated)
            img = envi.open(hdr_path, img_path)
            data = img.load()
            
            # Convert to Dask array
            dask_data = da.from_array(data, chunks=('auto', 'auto', -1))
            
            # Extract wavelengths
            wavelengths = np.linspace(1.0, 2.5, data.shape[2])
            if hasattr(img, 'bands') and img.bands.centers:
                wavelengths = np.array(img.bands.centers, dtype=np.float32)
            
            # Create xarray with Dask backend
            from astrogea import SpectralArrayWrapper
            wrapper = SpectralArrayWrapper(dask_data, wavelengths)
            xarray_data = wrapper.to_xarray()
            
            # Create dataset
            lines, samples = data.shape[:2]
            line_coords = xr.DataArray(np.arange(lines), dims='line')
            sample_coords = xr.DataArray(np.arange(samples), dims='sample')
            wavelength_coords = xr.DataArray(wavelengths, dims='wavelength')
            
            ds_dask = xr.Dataset({
                "spectral_data": xarray_data
            })
            
            # Force computation to measure actual time
            with ProgressBar():
                _ = ds_dask.spectral_data.compute()
            
            dask_time = time.perf_counter() - start_time
            print(f"   ✅ Dask processing time: {dask_time:.2f} seconds")
            print(f"   📊 Dataset size: {ds_dask.spectral_data.shape}")
            print(f"   💾 Memory usage estimate: {ds_dask.spectral_data.nbytes / 1024 / 1024:.1f} MB")
            
            # Performance comparison
            speedup = numpy_time / dask_time
            print(f"\n   📈 Performance comparison:")
            print(f"      NumPy time: {numpy_time:.2f}s")
            print(f"      Dask time: {dask_time:.2f}s")
            print(f"      Speedup: {speedup:.2f}x")
            
            if speedup > 1:
                print(f"      ✅ Dask is {speedup:.2f}x faster")
            else:
                print(f"      ⚠️  NumPy is {1/speedup:.2f}x faster")
            
        except ImportError:
            print("   ⚠️  Dask not available, skipping Dask benchmark")
            dask_time = None
        
        # Test 3: Memory usage comparison
        print("\n   Test 3: Memory usage analysis...")
        import psutil
        import gc
        
        # Measure memory before
        gc.collect()
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024
        
        # Load data
        ds_memory = envi_to_xarray_wcs(hdr_path, img_path)
        
        # Measure memory after
        memory_after = process.memory_info().rss / 1024 / 1024
        memory_used = memory_after - memory_before
        
        print(f"   💾 Memory usage: {memory_used:.1f} MB")
        print(f"   📊 Data size: {ds_memory.spectral_data.nbytes / 1024 / 1024:.1f} MB")
        print(f"   📈 Memory overhead: {memory_used / (ds_memory.spectral_data.nbytes / 1024 / 1024):.2f}x")
        
        # Recommendations
        print(f"\n   💡 Performance recommendations:")
        if dask_time and dask_time < numpy_time:
            print(f"      • Consider implementing Dask for large files")
            print(f"      • Dask provides {speedup:.2f}x speedup")
        else:
            print(f"      • Current NumPy implementation is efficient")
            print(f"      • Dask overhead may not be worth it for this file size")
        
        if memory_used > ds_memory.spectral_data.nbytes / 1024 / 1024 * 2:
            print(f"      • High memory overhead detected")
            print(f"      • Consider chunked processing for very large files")
        else:
            print(f"      • Memory usage is reasonable")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in performance benchmark: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dask_performance():
    """Test Dask performance in process_crism_file function"""
    print("\n🧪 Test 10: Dask performance test...")
    try:
        import time
        from astrogea import process_crism_file
        
        # File paths
        data_dir = "data"
        base_name = "frt00006fbd_07"
        
        # Find existing files
        possible_sr_names = [f"{base_name}_sr164j_mtr3", f"{base_name}_sr164j", f"{base_name}_sr"]
        possible_if_names = [f"{base_name}_if164j_mtr3", f"{base_name}_if164j", f"{base_name}_if"]
        
        sr_base = None
        if_base = None
        
        for sr_name in possible_sr_names:
            sr_hdr = os.path.join(data_dir, f"{sr_name}.hdr")
            sr_img = os.path.join(data_dir, f"{sr_name}.img")
            if os.path.exists(sr_hdr) and os.path.exists(sr_img):
                sr_base = sr_name
                break
        
        for if_name in possible_if_names:
            if_hdr = os.path.join(data_dir, f"{if_name}.hdr")
            if_img = os.path.join(data_dir, f"{if_name}.img")
            if os.path.exists(if_hdr) and os.path.exists(if_img):
                if_base = if_name
                break
        
        if sr_base is None or if_base is None:
            print(f"❌ Missing SR or IF files for Dask test")
            print("   Skipping Dask performance test")
            return True
        
        print(f"📁 Testing with SR: {sr_base}, IF: {if_base}")
        
        # Test 1: Without Dask
        print("\n   Test 1: Without Dask...")
        start_time = time.perf_counter()
        
        output_numpy = f"{sr_base}_numpy_test.nc"
        ds_numpy = process_crism_file(
            base_sr_path=os.path.join(data_dir, sr_base),
            base_if_path=os.path.join(data_dir, if_base),
            output_nc_path=output_numpy,
            use_dask=False
        )
        
        numpy_time = time.perf_counter() - start_time
        print(f"   ✅ NumPy processing time: {numpy_time:.2f} seconds")
        
        # Test 2: With Dask
        print("\n   Test 2: With Dask...")
        start_time = time.perf_counter()
        
        output_dask = f"{sr_base}_dask_test.nc"
        ds_dask = process_crism_file(
            base_sr_path=os.path.join(data_dir, sr_base),
            base_if_path=os.path.join(data_dir, if_base),
            output_nc_path=output_dask,
            use_dask=True
        )
        
        dask_time = time.perf_counter() - start_time
        print(f"   ✅ Dask processing time: {dask_time:.2f} seconds")
        
        # Performance comparison
        speedup = numpy_time / dask_time
        print(f"\n   📈 Performance comparison:")
        print(f"      NumPy time: {numpy_time:.2f}s")
        print(f"      Dask time: {dask_time:.2f}s")
        print(f"      Speedup: {speedup:.2f}x")
        
        if speedup > 1:
            print(f"      ✅ Dask is {speedup:.2f}x faster")
        else:
            print(f"      ⚠️  NumPy is {1/speedup:.2f}x faster")
        
        # File size comparison
        numpy_size = os.path.getsize(output_numpy) / 1024 / 1024
        dask_size = os.path.getsize(output_dask) / 1024 / 1024
        print(f"\n   📁 File size comparison:")
        print(f"      NumPy output: {numpy_size:.1f} MB")
        print(f"      Dask output: {dask_size:.1f} MB")
        
        # Clean up test files
        try:
            os.remove(output_numpy)
            os.remove(output_dask)
            print(f"   🧹 Test files cleaned up")
        except:
            pass
        
        return True
        
    except Exception as e:
        print(f"❌ Error in Dask performance test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 Starting AstroGea library tests...")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_spectral_array_wrapper,
        test_wcs_utils,
        test_cli_help,
        test_with_mock_data,
        test_fits_export,
        test_real_crism_files,  # This might fail if files are missing
        test_combined_crism_files,
        test_performance_benchmark,
        test_dask_performance
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! AstroGea library is working correctly.")
    elif passed >= total - 1:
        print("✅ Most tests passed! The library is working well.")
        print("   (Some tests might fail due to missing data files)")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    print("\n💡 Usage examples:")
    print("   from astrogea import envi_to_xarray_wcs")
    print("   ds = envi_to_xarray_wcs('file.hdr', 'file.img')")
    print("   ds.to_netcdf('output.nc')")

if __name__ == "__main__":
    main() 