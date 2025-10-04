"""
Kubernetes worker for astrogea distributed processing.
"""

import os
import sys
import argparse
import logging
import json
from typing import List, Dict, Any
from pathlib import Path

# Add astrogea to path
sys.path.insert(0, '/opt/astrogea')

from astrogea.config import create_config, Environment
from astrogea.storage import create_storage_manager
from astrogea.core import process_crism_file, continuum_removal, band_parameters_mafic

logger = logging.getLogger(__name__)

class AstrogeaWorker:
    """Worker class for processing astrogea jobs in Kubernetes."""
    
    def __init__(self, config_path: str = "/opt/astrogea/config.yaml"):
        self.config = create_config(Environment.KUBERNETES, config_path)
        self.storage_manager = create_storage_manager(self.config.get_storage_config())
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config.logging.level),
            format=self.config.logging.format
        )
        
        logger.info("Astrogea worker initialized")
        logger.info(f"Environment: {self.config.environment.value}")
        logger.info(f"Available storage backends: {self.config.get_available_storage_backends()}")
    
    def process_files(self, input_files: List[str], output_path: str, 
                     processing_type: str = "crism", **kwargs) -> Dict[str, Any]:
        """Process input files and save results."""
        logger.info(f"Processing {len(input_files)} files")
        logger.info(f"Processing type: {processing_type}")
        logger.info(f"Output path: {output_path}")
        
        results = []
        
        try:
            for i, input_file in enumerate(input_files):
                logger.info(f"Processing file {i+1}/{len(input_files)}: {input_file}")
                
                if processing_type == "crism":
                    result = self._process_crism_file(input_file, output_path, **kwargs)
                elif processing_type == "continuum":
                    result = self._process_continuum_removal(input_file, output_path, **kwargs)
                elif processing_type == "mafic":
                    result = self._process_mafic_analysis(input_file, output_path, **kwargs)
                else:
                    raise ValueError(f"Unknown processing type: {processing_type}")
                
                results.append(result)
                logger.info(f"Completed processing file {i+1}")
            
            return {
                'status': 'success',
                'processed_files': len(input_files),
                'results': results,
                'output_path': output_path
            }
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'processed_files': len(results),
                'results': results
            }
    
    def _process_crism_file(self, input_file: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """Process CRISM file."""
        try:
            # For CRISM processing, we expect two files: SR and IF
            if ',' in input_file:
                sr_file, if_file = input_file.split(',')
            else:
                # Assume single file processing
                sr_file = input_file
                if_file = None
            
            # Download files to local temporary directory
            temp_dir = Path("/tmp/astrogea_processing")
            temp_dir.mkdir(exist_ok=True)
            
            local_sr = temp_dir / "sr_data.hdr"
            local_if = temp_dir / "if_data.hdr" if if_file else None
            
            # Download SR file
            self.storage_manager.download_to_local(sr_file, str(local_sr))
            
            # Download IF file if provided
            if if_file and local_if:
                self.storage_manager.download_to_local(if_file, str(local_if))
            
            # Process file
            base_sr_path = str(local_sr).replace('.hdr', '')
            base_if_path = str(local_if).replace('.hdr', '') if local_if else None
            output_nc_path = temp_dir / "output.nc"
            
            if base_if_path:
                ds = process_crism_file(
                    base_sr_path, 
                    base_if_path, 
                    str(output_nc_path),
                    use_dask=self.config.processing.use_dask
                )
            else:
                # Single file processing
                from astrogea.core import envi_to_xarray_wcs
                ds = envi_to_xarray_wcs(str(local_sr))
                ds.to_netcdf(str(output_nc_path))
            
            # Upload result
            result_uri = f"{output_path}/result_{Path(input_file).stem}.nc"
            self.storage_manager.upload_from_local(str(output_nc_path), result_uri)
            
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir)
            
            return {
                'input_file': input_file,
                'output_file': result_uri,
                'status': 'success',
                'dataset_info': {
                    'shape': dict(ds.dims),
                    'variables': list(ds.data_vars.keys()),
                    'coordinates': list(ds.coords.keys())
                }
            }
            
        except Exception as e:
            logger.error(f"CRISM processing failed: {e}")
            return {
                'input_file': input_file,
                'status': 'error',
                'error': str(e)
            }
    
    def _process_continuum_removal(self, input_file: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """Process continuum removal."""
        try:
            # Load data
            data = self.storage_manager.read(input_file)
            
            # Convert to numpy array (simplified for example)
            import numpy as np
            import xarray as xr
            
            # Load as xarray dataset
            temp_file = "/tmp/temp_input.nc"
            with open(temp_file, 'wb') as f:
                f.write(data)
            
            ds = xr.open_dataset(temp_file)
            
            # Extract spectral data
            if 'spectral_data' in ds:
                img = ds.spectral_data.values
                wavelengths = ds.wavelength.values
            else:
                # Assume first data variable is spectral
                data_var = list(ds.data_vars.keys())[0]
                img = ds[data_var].values
                wavelengths = ds.wavelength.values if 'wavelength' in ds.coords else np.linspace(1.0, 2.5, img.shape[2])
            
            # Apply continuum removal
            MIN = kwargs.get('MIN', 1200)
            MAX = kwargs.get('MAX', 2200)
            
            result, x = continuum_removal(
                img, wavelengths, MIN, MAX,
                use_dask=self.config.processing.use_dask
            )
            
            # Create output dataset
            output_ds = xr.Dataset({
                'continuum_removed': (['line', 'sample', 'wavelength'], result)
            }, coords={
                'line': ds.line if 'line' in ds.coords else np.arange(result.shape[0]),
                'sample': ds.sample if 'sample' in ds.coords else np.arange(result.shape[1]),
                'wavelength': x
            })
            
            # Save result
            temp_output = "/tmp/temp_output.nc"
            output_ds.to_netcdf(temp_output)
            
            # Upload result
            result_uri = f"{output_path}/continuum_removed_{Path(input_file).stem}.nc"
            with open(temp_output, 'rb') as f:
                result_data = f.read()
            self.storage_manager.write(result_uri, result_data)
            
            # Cleanup
            os.unlink(temp_file)
            os.unlink(temp_output)
            
            return {
                'input_file': input_file,
                'output_file': result_uri,
                'status': 'success',
                'processing_params': {
                    'MIN': MIN,
                    'MAX': MAX,
                    'wavelength_range': f"{x.min():.1f}-{x.max():.1f}"
                }
            }
            
        except Exception as e:
            logger.error(f"Continuum removal failed: {e}")
            return {
                'input_file': input_file,
                'status': 'error',
                'error': str(e)
            }
    
    def _process_mafic_analysis(self, input_file: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """Process mafic analysis."""
        try:
            # Load continuum removed data
            data = self.storage_manager.read(input_file)
            
            # Convert to numpy array
            import numpy as np
            import xarray as xr
            
            temp_file = "/tmp/temp_input.nc"
            with open(temp_file, 'wb') as f:
                f.write(data)
            
            ds = xr.open_dataset(temp_file)
            
            # Extract continuum removed data
            if 'continuum_removed' in ds:
                img_removed = ds.continuum_removed.values
                wavelengths = ds.wavelength.values
            else:
                raise ValueError("No continuum_removed data found")
            
            # Apply mafic analysis
            nbands = kwargs.get('nbands', 5)
            windows_nm = kwargs.get('windows_nm', 75)
            resolution_nm = kwargs.get('resolution_nm', 5)
            tol = kwargs.get('tol', 10)
            
            mafic_map = band_parameters_mafic(
                img_removed, wavelengths,
                nbands=nbands, windows_nm=windows_nm,
                resolution_nm=resolution_nm, tol=tol,
                use_dask=self.config.processing.use_dask
            )
            
            # Create output dataset
            output_ds = xr.Dataset({
                'mafic_parameters': (['line', 'sample', 'parameter'], mafic_map)
            }, coords={
                'line': ds.line if 'line' in ds.coords else np.arange(mafic_map.shape[0]),
                'sample': ds.sample if 'sample' in ds.coords else np.arange(mafic_map.shape[1]),
                'parameter': [f'band_{i//5}_{param}' for i in range(0, nbands*5, 5) 
                             for param in ['minimum', 'center', 'depth', 'area', 'asymmetry']]
            })
            
            # Save result
            temp_output = "/tmp/temp_output.nc"
            output_ds.to_netcdf(temp_output)
            
            # Upload result
            result_uri = f"{output_path}/mafic_analysis_{Path(input_file).stem}.nc"
            with open(temp_output, 'rb') as f:
                result_data = f.read()
            self.storage_manager.write(result_uri, result_data)
            
            # Cleanup
            os.unlink(temp_file)
            os.unlink(temp_output)
            
            return {
                'input_file': input_file,
                'output_file': result_uri,
                'status': 'success',
                'processing_params': {
                    'nbands': nbands,
                    'windows_nm': windows_nm,
                    'resolution_nm': resolution_nm,
                    'tol': tol
                }
            }
            
        except Exception as e:
            logger.error(f"Mafic analysis failed: {e}")
            return {
                'input_file': input_file,
                'status': 'error',
                'error': str(e)
            }

def main():
    """Main worker entry point."""
    parser = argparse.ArgumentParser(description="Astrogea Kubernetes Worker")
    parser.add_argument("--input-files", required=True, help="Comma-separated input file URIs")
    parser.add_argument("--output", required=True, help="Output URI")
    parser.add_argument("--config", default="/opt/astrogea/config.yaml", help="Configuration file path")
    parser.add_argument("--processing-type", default="crism", 
                       choices=["crism", "continuum", "mafic"], help="Processing type")
    parser.add_argument("--params", help="JSON string of processing parameters")
    
    args = parser.parse_args()
    
    # Parse input files
    input_files = [f.strip() for f in args.input_files.split(',')]
    
    # Parse processing parameters
    processing_params = {}
    if args.params:
        try:
            processing_params = json.loads(args.params)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse parameters JSON: {e}")
            sys.exit(1)
    
    # Initialize worker
    worker = AstrogeaWorker(args.config)
    
    # Process files
    result = worker.process_files(
        input_files=input_files,
        output_path=args.output,
        processing_type=args.processing_type,
        **processing_params
    )
    
    # Output result
    print(json.dumps(result, indent=2))
    
    # Exit with appropriate code
    if result['status'] == 'success':
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
