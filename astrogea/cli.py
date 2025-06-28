import argparse
from .core import process_crism_file

def main():
    parser = argparse.ArgumentParser(description="Process CRISM data and save in NetCDF format.")
    parser.add_argument("--sr", required=True, help="Base path to the .hdr Surface Reflectance file (without extension)")
    parser.add_argument("--ifile", required=True, help="Base path to the .hdr Incidence Factor file (without extension)")
    parser.add_argument("--out", required=True, help="Output NetCDF file")
    parser.add_argument("--dask", action="store_true", help="Use Dask to load files")

    args = parser.parse_args()

    ds = process_crism_file(args.sr, args.ifile, args.out, use_dask=args.dask)
    print(f"File saved in: {args.out}")
    print(ds)
