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

        # Determine input format and produce output with dims ('line', 'sample', 'wavelength')
        if shape[0] == len(self.wavelengths):
            # Input format: (wavelength, line, sample) -> transpose to (line, sample, wavelength)
            _, rows, cols = shape
            data_out = np.transpose(self.data, (1, 2, 0))
        elif shape[2] == len(self.wavelengths):
            # Input format: (line, sample, wavelength) -> already correct
            rows, cols, _ = shape
            data_out = self.data
        elif shape[1] == len(self.wavelengths):
            # Input format: (line, wavelength, sample) -> transpose to (line, sample, wavelength)
            rows, _, cols = shape
            data_out = np.transpose(self.data, (0, 2, 1))
        else:
            raise ValueError(
                f"Cannot determine dimension order. Data shape: {shape}, wavelengths: {len(self.wavelengths)}"
            )

        dims = ['line', 'sample', self.wavelength_dim]
        coords = {
            'line': np.arange(rows),
            'sample': np.arange(cols),
            self.wavelength_dim: self.wavelengths,
        }
        return xr.DataArray(data_out, dims=dims, coords=coords)
