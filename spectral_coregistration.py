from scipy.interpolate import interp1d

def coregister_spectra(reference_wavelengths, new_wavelengths, new_reflectance):
    """
    Interpolates new reflectance values onto the reference wavelength grid.
    
    Parameters:
    - reference_wavelengths: Array of target wavelength values (from reference spectra)
    - new_wavelengths: Array of original wavelength values for new spectrum
    - new_reflectance: Array of reflectance values corresponding to new_wavelengths
    
    Returns:
    - Interpolated reflectance values matching reference_wavelengths
    """
    interp_func = interp1d(new_wavelengths, new_reflectance, kind='linear', fill_value="extrapolate")
    return interp_func(reference_wavelengths)
