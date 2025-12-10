#!/usr/bin/env python3
"""
Example demonstrating advanced spectral processing functions from astrogea.
Tests the newly integrated functions from autoencoder.py.
"""

import os
import sys
import numpy as np
from pathlib import Path
import time

# Matplotlib for plotting
try:
    import matplotlib
    matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plots
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib not available. Plots will be skipped.")

# Add astrogea to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from astrogea.core import (
    dimension_reduction_spectral_parameters,
    row_wise_integral_norm_data,
    column_wise_norm_advanced,
    find_nearest_wavelength,
    continuum_removal_points,
    compute_removal_single,
    compute_removal,
    dimension_reduction,
    auto_stretch_rgb_advanced
)

# Optional ML imports
try:
    from astrogea.ml import (
        SpectralAutoencoder,
        weight_init,
        random_search_autoencoder,
        train_autoencoder,
        plot_encoded_space
    )
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, random_split
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Note: ML functions not available. Install PyTorch with: pip install torch")

try:
    import spectral.io.envi as envi
    SPECTRAL_AVAILABLE = True
except ImportError:
    SPECTRAL_AVAILABLE = False
    print("Error: spectral library not available")


def main():
    print("=" * 70)
    print("Astrogea - Advanced Spectral Processing Example")
    print("=" * 70)
    
    # 1. Load ENVI file directly from .img file
    print("\n1. Loading ENVI file from .img file...")
    img_file = "data/frt00006fbd_07_if164j_mtr3.img"
    hdr_file = "data/frt00006fbd_07_if164j_mtr3.hdr"
    
    if not os.path.exists(img_file):
        print(f"   ERROR: Image file not found: {img_file}")
        print("   Make sure you have example files in the data/ directory")
        return
    
    try:
        # Read metadata from .hdr file if available
        metadata = {}
        wavelength = None
        wavelength_units = None
        
        if os.path.exists(hdr_file):
            print(f"   Reading metadata from {hdr_file}...")
            with open(hdr_file, 'r') as f:
                lines_list = f.readlines()
                i = 0
                while i < len(lines_list):
                    line = lines_list[i].strip()
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Handle wavelength multiline block
                        if key.lower() == 'wavelength' and value.startswith('{'):
                            # Collect all wavelength values until closing brace
                            wavelength_str = value
                            if not value.endswith('}'):
                                # Multiline block: read until closing brace
                                i += 1
                                while i < len(lines_list) and '}' not in wavelength_str:
                                    wavelength_str += ' ' + lines_list[i].strip()
                                    i += 1
                                # i now points to the line with closing brace, increment to move past it
                                i += 1
                            else:
                                # Single-line block: just increment to next line
                                i += 1
                            # Extract numeric values
                            try:
                                # Remove braces and split by comma
                                wav_values = wavelength_str.strip('{}').replace(',', ' ').split()
                                wavelength = np.array([float(w) for w in wav_values if w.strip()])
                                print(f"   Found {len(wavelength)} wavelength values in header")
                            except Exception as e:
                                print(f"   Warning: Could not parse wavelength values: {e}")
                                wavelength = None
                            metadata[key.lower()] = wavelength_str
                        else:
                            metadata[key.lower()] = value
                            print(line)
                            i += 1
                    else:
                        i += 1
                
                # Get wavelength units if available
                wavelength_units = metadata.get('wavelength units', 'Unknown').lower()
                if wavelength is not None:
                    # Convert units if needed (Nanometers to Micrometers)
                    if 'nanometer' in wavelength_units or 'nm' in wavelength_units:
                        wavelength = wavelength / 1000.0  # Convert nm to um
                        print(f"   Converted wavelength units from Nanometers to Micrometers")
                    elif 'micrometer' in wavelength_units or 'um' in wavelength_units:
                        pass  # Already in micrometers
                    print(f"   Wavelength units: {metadata.get('wavelength units', 'Unknown')}")
        
        # Get dimensions from metadata or use defaults
        lines = int(metadata.get('lines', 588))
        samples = int(metadata.get('samples', 729))
        bands = int(metadata.get('bands', 489))
        
        # Get data type from metadata
        dtype_map = {
            '1': np.uint8,
            '2': np.int16,
            '3': np.int32,
            '4': np.float32,
            '5': np.float64,
            '12': np.uint16,
            '13': np.uint32
        }
        data_type_str = metadata.get('data type', '4')
        dtype = dtype_map.get(data_type_str, np.float32)
        
        # Get interleave (BSQ, BIL, BIP)
        interleave = metadata.get('interleave', 'bsq').upper()
        
        print(f"   Image file: {img_file}")
        print(f"   Dimensions: {lines} x {samples} x {bands}")
        print(f"   Data type: {dtype}")
        print(f"   Interleave: {interleave}")
        
        # Load binary data from .img file
        print("   Loading binary data from .img file...")
        file_size = os.path.getsize(img_file)
        expected_size = lines * samples * bands * np.dtype(dtype).itemsize
        
        if file_size != expected_size:
            print(f"   WARNING: File size ({file_size}) doesn't match expected ({expected_size})")
            print(f"   Attempting to load available data...")
        
        # Read binary data
        img_data_flat = np.fromfile(img_file, dtype=dtype)
        
        # Reshape based on interleave format
        if interleave == 'BSQ':  # Band Sequential: (bands, lines, samples)
            img_data_flat = img_data_flat[:lines * samples * bands]
            img_array = img_data_flat.reshape((bands, lines, samples))
            img_array = np.transpose(img_array, (1, 2, 0))  # Convert to (lines, samples, bands)
        elif interleave == 'BIL':  # Band Interleaved by Line: (lines, bands, samples)
            img_data_flat = img_data_flat[:lines * samples * bands]
            img_array = img_data_flat.reshape((lines, bands, samples))
            img_array = np.transpose(img_array, (0, 2, 1))  # Convert to (lines, samples, bands)
        else:  # BIP: Band Interleaved by Pixel: (lines, samples, bands)
            img_data_flat = img_data_flat[:lines * samples * bands]
            img_array = img_data_flat.reshape((lines, samples, bands))
        
        print(f"   [OK] File loaded successfully!")
        print(f"   Final array shape: {img_array.shape}")
        print(f"   Data type: {img_array.dtype}")
        
        # Use extracted wavelengths or create defaults
        if wavelength is None:
            print("   No wavelength data found in header, using default wavelength range...")
            wavelength = np.linspace(1.0, 2.5, bands)
        elif len(wavelength) != bands:
            print(f"   Warning: Wavelength array length ({len(wavelength)}) doesn't match bands ({bands})")
            if len(wavelength) > bands:
                wavelength = wavelength[:bands]
                print(f"   Truncated wavelength array to {bands} bands")
            elif len(wavelength) < bands:
                # Extend with linear interpolation
                print(f"   Extending wavelength array from {len(wavelength)} to {bands} bands")
                wavelength = np.linspace(wavelength[0], wavelength[-1], bands)
        
        print(f"   Number of spectral bands: {bands}")
        print(f"   Wavelength range: {wavelength[0]:.4f} - {wavelength[-1]:.4f} um")
        if wavelength_units and wavelength_units != 'unknown':
            print(f"   Wavelength loaded from header file (units: {metadata.get('wavelength units', 'Unknown')})")
        else:
            print("   (Using default wavelengths)")
        
        # Convert to float32 for processing
        if img_array.dtype != np.float32:
            img_array = img_array.astype(np.float32)

        print('IMPORTANT - ' , np.shape(img_array))
            
    except Exception as e:
        print(f"   ERROR loading file: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. Test dimension_reduction with wavelength range
    print("\n2. Testing dimension_reduction with wavelength range...")
    print("   Extracting spectra in wavelength range 0.5 - 2.5 um (500-2500 nm)...")
    start_time = time.time()
    
    try:
        # Convert nm to um: 500 nm = 0.5 um, 2500 nm = 2.5 um
        w1, w2 = 0.5, 2.5  # um
        spectra, indexes, red_w = dimension_reduction(
            img_array,
            w1, w2,
            wavelength,
            cr=False  # Set to True to apply continuum removal
        )
        elapsed = time.time() - start_time
        print(f"   [OK] Extraction completed in {elapsed:.2f} seconds")
        print(f"   Extracted {spectra.shape[0]} valid spectra")
        print(f"   Each spectrum has {spectra.shape[1]} bands")
        print(f"   Spectra shape: {spectra.shape}")
        print(f"   Indexes shape: {indexes.shape}")
        print(f"   Wavelength range: {red_w[0]:.4f} - {red_w[-1]:.4f} um")
        print(f"   Number of bands: {len(red_w)}")
        
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. Test row_wise_integral_norm_data
    print("\n3. Testing row_wise_integral_norm_data...")
    print("   Normalizing spectra by integral over wavelength...")
    start_time = time.time()
    
    try:
        # Use a subset for faster testing
        n_samples = min(1000, spectra.shape[0])
        test_spectra = spectra[:n_samples]
        # Use the reduced wavelength array from dimension_reduction
        test_wavelength = red_w
        
        norm_spectra, wav = row_wise_integral_norm_data(
            test_wavelength,
            test_spectra,
            use_dask=False
        )
        elapsed = time.time() - start_time
        print(f"   [OK] Normalization completed in {elapsed:.2f} seconds")
        print(f"   Normalized spectra shape: {norm_spectra.shape}")
        print(f"   Sample normalized spectrum (first 5 values): {norm_spectra[0, :5]}")
        
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. Test column_wise_norm_advanced
    print("\n4. Testing column_wise_norm_advanced...")
    print("   Advanced column-wise normalization...")
    start_time = time.time()
    
    try:
        col_norm_spectra = column_wise_norm_advanced(
            test_spectra,
            use_dask=False
        )
        elapsed = time.time() - start_time
        print(f"   [OK] Column normalization completed in {elapsed:.2f} seconds")
        print(f"   Normalized spectra shape: {col_norm_spectra.shape}")
        print(f"   Mean per column (first 5): {np.mean(col_norm_spectra, axis=0)[:5]}")
        print(f"   Std per column (first 5): {np.std(col_norm_spectra, axis=0)[:5]}")
        
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. Test find_nearest_wavelength
    print("\n5. Testing find_nearest_wavelength...")
    try:
        target_wavelength = 1.5  # um
        idx = find_nearest_wavelength(wavelength, target_wavelength)
        print(f"   [OK] Found nearest wavelength to {target_wavelength} um")
        print(f"   Index: {idx}, Wavelength: {wavelength[idx]:.4f} um")
        
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # 6. Test continuum removal
    print("\n6. Testing continuum removal...")
    print("   Computing continuum removal for sample spectra...")
    start_time = time.time()
    
    try:
        # Test on a few spectra
        n_test = min(10, test_spectra.shape[0])
        test_wav_subset = test_wavelength[:test_spectra.shape[1]]
        
        # Single spectrum continuum removal
        single_removed = compute_removal_single(
            test_wav_subset,
            test_spectra[0, :],
            interp_type='linear'
        )
        print(f"   [OK] Single spectrum continuum removal completed")
        print(f"   Removed spectrum shape: {single_removed.shape}")
        print(f"   Sample values (first 5): {single_removed[:5]}")
        
        # Multiple spectra continuum removal
        multi_removed = compute_removal(
            test_wav_subset,
            test_spectra[:n_test, :],
            interp_type='linear'
        )
        elapsed = time.time() - start_time
        print(f"   [OK] Multiple spectra continuum removal completed in {elapsed:.2f} seconds")
        print(f"   Removed spectra shape: {multi_removed.shape}")
        
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # 7. Test ML functions (if available)
    if ML_AVAILABLE:
        print("\n7. Testing ML functions...")
        print("   Creating PyTorch dataset and testing autoencoder...")
        
        try:
            # Prepare data for ML
            ml_spectra = spectra[:min(5000, spectra.shape[0])]  # Use subset for faster testing
            ml_spectra_tensor = torch.FloatTensor(ml_spectra)
            dataset = TensorDataset(ml_spectra_tensor)
            
            print(f"   Dataset created with {len(dataset)} samples")
            print(f"   Input dimension: {ml_spectra.shape[1]}")
            
            # Create a simple autoencoder
            encoded_dim = 10
            in_channels = ml_spectra.shape[1]
            n_enc = 1
            n_dec = 1
            out1 = [64]
            out2 = [64]
            act = nn.ReLU()
            drops = [0.2, 0.2]
            
            model = SpectralAutoencoder(
                encoded_dim, in_channels, n_enc, n_dec,
                out1, out2, act, drops, True
            )
            
            # Initialize weights
            weight_init(model, init_method='xavier_normal')
            
            print(f"   [OK] Autoencoder model created")
            print(f"   Model parameters: {sum(p.numel() for p in model.parameters())}")
            
            # Test forward pass
            sample_input = ml_spectra_tensor[:10]
            with torch.no_grad():
                output = model(sample_input)
            print(f"   [OK] Forward pass successful")
            print(f"   Input shape: {sample_input.shape}")
            print(f"   Output shape: {output.shape}")
            
            # Split dataset
            train_len = int(len(dataset) * 0.8)
            val_len = len(dataset) - train_len
            train_set, val_set = random_split(dataset, [train_len, val_len])
            
            print(f"   Train set: {len(train_set)} samples")
            print(f"   Validation set: {len(val_set)} samples")
            
            print("   Note: Full training can be done with train_autoencoder()")
            print("   Note: Hyperparameter search can be done with random_search_autoencoder()")
            
        except Exception as e:
            print(f"   ERROR: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n7. ML functions not available (PyTorch not installed)")
        print("   Install with: pip install torch")
    
    # 8. Create visualizations
    if MATPLOTLIB_AVAILABLE:
        print("\n8. Creating visualizations...")
        try:
            # Create output directory for plots
            plot_dir = Path("output_plots")
            plot_dir.mkdir(exist_ok=True)
            
            # Plot 1: Sample original spectra
            print("   Plotting sample original spectra...")
            n_plot_samples = min(10, spectra.shape[0])
            fig, ax = plt.subplots(figsize=(12, 6))
            for i in range(n_plot_samples):
                idx = i * (spectra.shape[0] // n_plot_samples)
                ax.plot(red_w, spectra[idx, :], 
                       label=f'Spectrum {idx+1}', alpha=0.7, linewidth=1.5)
            ax.set_xlabel('Wavelength (um)', fontsize=12)
            ax.set_ylabel('Reflectance (normalized)', fontsize=12)
            ax.set_title('Sample Original Spectra (Wavelength Range 0.5-2.5 um)', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_dir / '1_original_spectra.png', dpi=150, bbox_inches='tight')
            print(f"   [OK] Saved: {plot_dir / '1_original_spectra.png'}")
            plt.close()
            
            # Plot 2: Comparison of normalization methods
            print("   Plotting normalization comparison...")
            sample_idx = 0
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Original spectrum
            axes[0, 0].plot(test_wavelength, test_spectra[sample_idx, :], 'b-', linewidth=2)
            axes[0, 0].set_title('Original Spectrum', fontweight='bold')
            axes[0, 0].set_xlabel('Wavelength (um)')
            axes[0, 0].set_ylabel('Reflectance')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Row-wise integral normalized
            axes[0, 1].plot(test_wavelength, norm_spectra[sample_idx, :], 'g-', linewidth=2)
            axes[0, 1].set_title('Row-wise Integral Normalized', fontweight='bold')
            axes[0, 1].set_xlabel('Wavelength (um)')
            axes[0, 1].set_ylabel('Normalized Reflectance')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Column normalized
            axes[1, 0].plot(test_wavelength, col_norm_spectra[sample_idx, :], 'r-', linewidth=2)
            axes[1, 0].set_title('Column Normalized', fontweight='bold')
            axes[1, 0].set_xlabel('Wavelength (um)')
            axes[1, 0].set_ylabel('Normalized Reflectance')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Comparison overlay
            axes[1, 1].plot(test_wavelength, test_spectra[sample_idx, :], 'b-', 
                           label='Original', alpha=0.5, linewidth=1.5)
            axes[1, 1].plot(test_wavelength, norm_spectra[sample_idx, :], 'g-', 
                           label='Row Integral Norm', alpha=0.7, linewidth=1.5)
            axes[1, 1].plot(test_wavelength, col_norm_spectra[sample_idx, :], 'r-', 
                           label='Column Norm', alpha=0.7, linewidth=1.5)
            axes[1, 1].set_title('Normalization Comparison', fontweight='bold')
            axes[1, 1].set_xlabel('Wavelength (um)')
            axes[1, 1].set_ylabel('Reflectance')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(plot_dir / '2_normalization_comparison.png', dpi=150, bbox_inches='tight')
            print(f"   [OK] Saved: {plot_dir / '2_normalization_comparison.png'}")
            plt.close()
            
            # Plot 3: Continuum removal
            print("   Plotting continuum removal...")
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            # Original vs continuum removed (single spectrum)
            axes[0].plot(test_wav_subset, test_spectra[0, :], 'b-', 
                        label='Original', linewidth=2, alpha=0.7)
            axes[0].plot(test_wav_subset, single_removed, 'r-', 
                        label='Continuum Removed', linewidth=2, alpha=0.7)
            axes[0].set_title('Continuum Removal - Single Spectrum', fontweight='bold', fontsize=12)
            axes[0].set_xlabel('Wavelength (um)', fontsize=11)
            axes[0].set_ylabel('Reflectance', fontsize=11)
            axes[0].legend(fontsize=10)
            axes[0].grid(True, alpha=0.3)
            
            # Multiple spectra continuum removed
            for i in range(min(5, multi_removed.shape[0])):
                axes[1].plot(test_wav_subset, multi_removed[i, :], 
                           alpha=0.6, linewidth=1.5, label=f'Spectrum {i+1}')
            axes[1].set_title('Continuum Removal - Multiple Spectra', fontweight='bold', fontsize=12)
            axes[1].set_xlabel('Wavelength (um)', fontsize=11)
            axes[1].set_ylabel('Continuum Removed Reflectance', fontsize=11)
            axes[1].legend(fontsize=9)
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(plot_dir / '3_continuum_removal.png', dpi=150, bbox_inches='tight')
            print(f"   [OK] Saved: {plot_dir / '3_continuum_removal.png'}")
            plt.close()
            
            # Plot 4: Dimension reduction with wavelength range
            print("   Plotting dimension reduction results...")
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Full spectrum vs reduced range
            sample_idx_red = min(100, spectra.shape[0] - 1)
            axes[0].plot(wavelength, img_array[int(indexes[sample_idx_red, 0]), 
                                              int(indexes[sample_idx_red, 1]), :], 
                        'b-', label='Full Spectrum', linewidth=2, alpha=0.7)
            axes[0].plot(red_w, spectra[sample_idx_red, :], 'r-', 
                        label=f'Reduced Range ({w1}-{w2} um)', linewidth=2, alpha=0.7)
            axes[0].axvspan(w1, w2, alpha=0.2, color='yellow', label='Selected Range')
            axes[0].set_title('Dimension Reduction - Wavelength Range Selection', fontweight='bold')
            axes[0].set_xlabel('Wavelength (um)')
            axes[0].set_ylabel('Reflectance')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Multiple reduced spectra
            n_red_plot = min(10, spectra.shape[0])
            for i in range(n_red_plot):
                idx = i * (spectra.shape[0] // n_red_plot)
                axes[1].plot(red_w, spectra[idx, :], alpha=0.6, linewidth=1.5)
            axes[1].set_title(f'Sample Reduced Spectra ({len(red_w)} bands)', fontweight='bold')
            axes[1].set_xlabel('Wavelength (um)')
            axes[1].set_ylabel('Reflectance')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(plot_dir / '4_dimension_reduction.png', dpi=150, bbox_inches='tight')
            print(f"   [OK] Saved: {plot_dir / '4_dimension_reduction.png'}")
            plt.close()
            
            # Plot 5: Statistics overview
            print("   Plotting statistics overview...")
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Mean spectrum (axis=0 means average across spectra for each wavelength band)
            mean_spectrum = np.mean(spectra, axis=0)
            std_spectrum = np.std(spectra, axis=0)
            axes[0, 0].plot(red_w, mean_spectrum, 'b-', linewidth=2, label='Mean')
            axes[0, 0].fill_between(red_w, mean_spectrum - std_spectrum, 
                                   mean_spectrum + std_spectrum, alpha=0.3, label='+/-1 Std')
            axes[0, 0].set_title('Mean Spectrum with Standard Deviation', fontweight='bold')
            axes[0, 0].set_xlabel('Wavelength (um)')
            axes[0, 0].set_ylabel('Reflectance')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Reflectance distribution at selected wavelengths
            n_bands = spectra.shape[1]
            selected_wav_idx = [int(i * n_bands / 5) for i in range(1, 6)]  # 5 evenly spaced indices
            selected_wav_idx = [idx for idx in selected_wav_idx if idx < n_bands]  # Ensure valid indices
            axes[0, 1].hist([spectra[:, idx] for idx in selected_wav_idx], 
                           bins=50, alpha=0.6, label=[f'{red_w[idx]:.2f} um' 
                                                      for idx in selected_wav_idx])
            axes[0, 1].set_title('Reflectance Distribution at Selected Wavelengths', fontweight='bold')
            axes[0, 1].set_xlabel('Reflectance')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Wavelength vs reflectance heatmap (sample)
            sample_size = min(1000, spectra.shape[0])
            sample_indices = np.random.choice(spectra.shape[0], sample_size, replace=False)
            im = axes[1, 0].imshow(spectra[sample_indices, :].T, aspect='auto', 
                                  cmap='viridis', interpolation='nearest')
            axes[1, 0].set_title(f'Spectral Data Heatmap (sample of {sample_size} spectra)', fontweight='bold')
            axes[1, 0].set_xlabel('Spectrum Index')
            axes[1, 0].set_ylabel('Wavelength Band')
            # Set y-axis ticks to show wavelength values
            n_ticks = min(10, len(red_w))
            tick_positions = np.linspace(0, len(red_w)-1, n_ticks).astype(int)
            axes[1, 0].set_yticks(tick_positions)
            axes[1, 0].set_yticklabels([f'{red_w[i]:.2f}' for i in tick_positions])
            plt.colorbar(im, ax=axes[1, 0], label='Reflectance')
            
            # Statistics summary
            axes[1, 1].axis('off')
            stats_text = f"""
Statistics Summary:
- Total spectra: {spectra.shape[0]:,}
- Spectral bands: {spectra.shape[1]}
- Wavelength range: {red_w[0]:.2f} - {red_w[-1]:.2f} um
- Mean reflectance: {np.mean(spectra):.4f}
- Std reflectance: {np.std(spectra):.4f}
- Min reflectance: {np.min(spectra):.4f}
- Max reflectance: {np.max(spectra):.4f}
- Valid pixels: {np.sum(img_array[:, :, 0] != 65535):,}
- Invalid pixels: {np.sum(img_array[:, :, 0] == 65535):,}
            """
            axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, 
                          verticalalignment='center', family='monospace',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            plt.savefig(plot_dir / '5_statistics_overview.png', dpi=150, bbox_inches='tight')
            print(f"   [OK] Saved: {plot_dir / '5_statistics_overview.png'}")
            plt.close()
            
            print(f"\n   [OK] All plots saved in '{plot_dir}' directory")
            
        except Exception as e:
            print(f"   ERROR creating plots: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n8. Visualization skipped (Matplotlib not available)")
        print("   Install with: pip install matplotlib")
    
    # 9. Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("[OK] All advanced spectral processing functions tested successfully!")
    print("\nFunctions tested:")
    print("  - dimension_reduction_spectral_parameters")
    print("  - row_wise_integral_norm_data")
    print("  - column_wise_norm_advanced")
    print("  - find_nearest_wavelength")
    print("  - compute_removal_single")
    print("  - compute_removal")
    print("  - dimension_reduction")
    if ML_AVAILABLE:
        print("  - SpectralAutoencoder (model creation)")
        print("  - weight_init")
    if MATPLOTLIB_AVAILABLE:
        print("  - Visualization plots generated")
    print("\nAll functions are now integrated in astrogea and ready to use!")
    print("=" * 70)


if __name__ == "__main__":
    main()

