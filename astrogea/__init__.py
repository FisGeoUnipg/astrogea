from .core import (
    process_crism_file, envi_to_xarray_wcs, band_parameters_mafic, continuum_removal, 
    smoothing_moving_average, coregister_spectra, remove_crism_bad_ranges_cube, 
    row_norm, column_norm, center_norm, L1_norm, minmax, robust_scaler, derivative, 
    log_1_r_norm, baseline_correction_cube, auto_stretch_rgb, unison_shuffled_copies, 
    merge_datacubes, spetial_merge_datacubes, hypermerge_spatial,
    # Advanced spectral processing functions
    row_wise_integral_norm_data, column_wise_norm_advanced, find_nearest_wavelength,
    continuum_removal_points, compute_removal_single, compute_removal,
    dimension_reduction, dimension_reduction_spectral_parameters, auto_stretch_rgb_advanced
)
from .spectral_wrapper import SpectralArrayWrapper
from .wcs_utils import parse_envi_map_info_list, create_wcs_from_parsed_info, create_wcs_header_dict

# ML functions (optional - require PyTorch)
try:
    from .ml import (
        SpectralAutoencoder, Net, GumbelSoftmax, weight_init, random_search_autoencoder,
        RandomSearch_autoencoder, train_autoencoder, train_cnn, plot_encoded_space, plot_n_encoded,
        INITS
    )
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

__version__ = "0.1.15"

__all__ = [
    "process_crism_file",
    "envi_to_xarray_wcs",
    "SpectralArrayWrapper",
    "parse_envi_map_info_list",
    "create_wcs_from_parsed_info",
    "create_wcs_header_dict",
    "band_parameters_mafic",
    "continuum_removal",
    "smoothing_moving_average",
    "coregister_spectra",
    "remove_crism_bad_ranges_cube",
    "row_norm",
    "column_norm",
    "center_norm",
    "L1_norm",
    "minmax",
    "robust_scaler",
    "derivative",
    "log_1_r_norm",
    "baseline_correction_cube",
    "auto_stretch_rgb",
    "unison_shuffled_copies",
    "merge_datacubes",
    "spetial_merge_datacubes",
    "hypermerge_spatial",
    # Advanced spectral processing
    "row_wise_integral_norm_data",
    "column_wise_norm_advanced",
    "find_nearest_wavelength",
    "continuum_removal_points",
    "compute_removal_single",
    "compute_removal",
    "dimension_reduction",
    "dimension_reduction_spectral_parameters",
    "auto_stretch_rgb_advanced",
]

# Add ML functions if available
if ML_AVAILABLE:
    __all__.extend([
        "SpectralAutoencoder",
        "Net",
        "GumbelSoftmax",
        "weight_init",
        "random_search_autoencoder",
        "RandomSearch_autoencoder",
        "train_autoencoder",
        "train_cnn",
        "plot_encoded_space",
        "plot_n_encoded",
        "INITS",
    ])
