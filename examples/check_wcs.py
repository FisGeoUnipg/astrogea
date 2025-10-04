#!/usr/bin/env python3
"""
Verifica se il file NetCDF contiene informazioni WCS.
"""

import os
import sys
from pathlib import Path
import xarray as xr

# Aggiungi astrogea al path
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_wcs_info(file_path):
    """Verifica informazioni WCS nel file NetCDF."""
    
    if not os.path.exists(file_path):
        print(f"❌ File non trovato: {file_path}")
        return
    
    print(f"=== Analisi WCS: {file_path} ===")
    
    try:
        # Carica dataset
        ds = xr.open_dataset(file_path)
        
        print(f"\n📊 Informazioni Dataset:")
        print(f"   Dimensioni: {ds.dims}")
        print(f"   Variabili: {list(ds.data_vars.keys())}")
        print(f"   Coordinate: {list(ds.coords.keys())}")
        
        print(f"\n🌍 Informazioni WCS:")
        
        # Controlla flag WCS
        has_wcs = ds.attrs.get('has_wcs', 0)
        print(f"   Flag WCS: {has_wcs}")
        
        if has_wcs == 1:
            print("   ✅ WCS disponibile!")
            
            # Mostra header WCS
            wcs_header = ds.attrs.get('wcs_header_dict', 'N/A')
            print(f"   WCS Header: {wcs_header}")
            
            # Mostra commento
            wcs_comment = ds.attrs.get('has_wcs_comment', 'N/A')
            print(f"   Commento: {wcs_comment}")
            
        else:
            print("   ⚠️  WCS non disponibile")
            wcs_comment = ds.attrs.get('has_wcs_comment', 'N/A')
            print(f"   Motivo: {wcs_comment}")
        
        print(f"\n📋 Tutti gli Attributi:")
        for key, value in ds.attrs.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"   {key}: {value[:100]}...")
            else:
                print(f"   {key}: {value}")
        
        # Informazioni coordinate
        print(f"\n🗺️  Coordinate Disponibili:")
        for coord_name, coord_data in ds.coords.items():
            print(f"   {coord_name}:")
            print(f"     - Shape: {coord_data.shape}")
            print(f"     - Range: {coord_data.min().values:.3f} - {coord_data.max().values:.3f}")
            if hasattr(coord_data, 'attrs'):
                print(f"     - Units: {coord_data.attrs.get('units', 'N/A')}")
                print(f"     - Long name: {coord_data.attrs.get('long_name', 'N/A')}")
        
        ds.close()
        
    except Exception as e:
        print(f"❌ Errore durante analisi: {e}")

def main():
    print("=== Verifica WCS in File NetCDF ===")
    
    # Controlla file di output dell'esempio
    output_file = "output/result.nc"
    
    if os.path.exists(output_file):
        check_wcs_info(output_file)
    else:
        print(f"❌ File di output non trovato: {output_file}")
        print("Esegui prima: python examples/simple_example.py")
        
        # Controlla se ci sono altri file NetCDF
        nc_files = list(Path(".").glob("**/*.nc"))
        if nc_files:
            print(f"\n📁 File NetCDF trovati:")
            for nc_file in nc_files:
                print(f"   {nc_file}")
                check_wcs_info(str(nc_file))
                print()

if __name__ == "__main__":
    main()













