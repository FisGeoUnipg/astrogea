import xarray as xr

loaded_data = xr.open_dataarray("frt000144ff_07_sr164j_mtr3_xarray.nc")
print(loaded_data)