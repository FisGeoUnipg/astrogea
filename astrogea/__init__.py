from .core import process_crism_file, envi_to_xarray_wcs, band_parameters_mafic, continuum_removal, smoothing_moving_average, coregister_spectra
from .spectral_wrapper import SpectralArrayWrapper
from .wcs_utils import parse_envi_map_info_list, create_wcs_from_parsed_info, create_wcs_header_dict

__version__ = "0.1.1"

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
    "coregister_spectra"
]
