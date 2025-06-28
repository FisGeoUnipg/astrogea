import numpy as np
import xarray as xr

class SpectralArrayWrapper:
    def __init__(self, data, wavelengths, wavelength_dim='wavelength'):
        self.data = data
        self.wavelengths = wavelengths
        self.wavelength_dim = wavelength_dim

    def to_xarray(self):
        shape = self.data.shape
        if len(shape) != 3:
            raise ValueError("Data must be 3D (bands, rows, cols)")

        # Determine dimension order
        if shape[0] == len(self.wavelengths):
            # Format: (wavelength, y, x)
            bands, rows, cols = shape
            dims = [self.wavelength_dim, 'y', 'x']
            coords = {self.wavelength_dim: self.wavelengths}
        elif shape[2] == len(self.wavelengths):
            # Format: (y, x, wavelength) - CRISM case
            rows, cols, bands = shape
            # Transpose to get (wavelength, y, x)
            data_transposed = np.transpose(self.data, (2, 0, 1))
            dims = [self.wavelength_dim, 'y', 'x']
            coords = {self.wavelength_dim: self.wavelengths}
            return xr.DataArray(
                data_transposed,
                dims=dims,
                coords=coords
            )
        elif shape[1] == len(self.wavelengths):
            # Format: (y, wavelength, x)
            rows, bands, cols = shape
            # Transpose to get (wavelength, y, x)
            data_transposed = np.transpose(self.data, (1, 0, 2))
            dims = [self.wavelength_dim, 'y', 'x']
            coords = {self.wavelength_dim: self.wavelengths}
            return xr.DataArray(
                data_transposed,
                dims=dims,
                coords=coords
            )
        else:
            raise ValueError(f"Cannot determine dimension order. Data shape: {shape}, wavelengths: {len(self.wavelengths)}")

        return xr.DataArray(
            self.data,
            dims=dims,
            coords=coords
        )
