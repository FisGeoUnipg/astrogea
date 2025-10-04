#!/usr/bin/env python3
"""
Esempio di esportazione in formato ASCII con coordinate WCS.
"""

import os
import sys
from pathlib import Path

# Aggiungi astrogea al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from astrogea.config import create_config, Environment
from astrogea.core import process_crism_file
from astrogea.export import export_to_ascii_wcs, export_to_ascii_spectral, export_to_csv_wcs

def main():
    print("=== Astrogea - Export ASCII WCS ===")
    
    # 1. Configurazione
    print("\n1. Configurazione...")
    config = create_config(Environment.LOCAL)
    print(f"   Ambiente: {config.environment.value}")
    
    # 2. File di input
    print("\n2. Preparazione file di input...")
    input_sr = "data/frt00006fbd_07_sr164j_mtr3.hdr"
    input_if = "data/frt00006fbd_07_if164j_mtr3.hdr"
    netcdf_output = "output/result.nc"
    
    # Verifica file di input
    if not os.path.exists(input_sr) or not os.path.exists(input_if):
        print("   ❌ File di input non trovati!")
        print("   Esegui prima: python examples/simple_example.py")
        return
    
    # 3. Elaborazione (se necessario)
    print("\n3. Elaborazione file CRISM...")
    if not os.path.exists(netcdf_output):
        print("   Elaborazione file...")
        os.makedirs("output", exist_ok=True)
        
        ds = process_crism_file(
            base_sr_path=input_sr.replace('.hdr', ''),
            base_if_path=input_if.replace('.hdr', ''),
            output_nc_path=netcdf_output,
            use_dask=True
        )
        print("   ✅ Elaborazione completata!")
    else:
        print("   ✅ File NetCDF già esistente!")
        import xarray as xr
        ds = xr.open_dataset(netcdf_output)
    
    # 4. Export ASCII WCS
    print("\n4. Export ASCII WCS...")
    
    # Export 1: Dati spettrali con coordinate
    ascii_output1 = "output/spectral_data_wcs.txt"
    try:
        export_to_ascii_wcs(
            dataset=ds,
            output_path=ascii_output1,
            variable='spectral_data',
            wavelength_idx=10,  # Wavelength specifica
            include_coordinates=True
        )
        print(f"   ✅ Export ASCII WCS: {ascii_output1}")
    except Exception as e:
        print(f"   ❌ Errore export ASCII WCS: {e}")
    
    # Export 2: Spettro di un pixel specifico
    ascii_output2 = "output/pixel_spectrum.txt"
    try:
        export_to_ascii_spectral(
            dataset=ds,
            output_path=ascii_output2,
            variable='spectral_data',
            pixel_coords=(50, 50)  # Pixel centrale
        )
        print(f"   ✅ Export spettro pixel: {ascii_output2}")
    except Exception as e:
        print(f"   ❌ Errore export spettro: {e}")
    
    # Export 3: CSV con WCS
    csv_output = "output/spectral_data_wcs.csv"
    try:
        export_to_csv_wcs(
            dataset=ds,
            output_path=csv_output,
            variable='spectral_data',
            include_metadata=True
        )
        print(f"   ✅ Export CSV WCS: {csv_output}")
    except Exception as e:
        print(f"   ❌ Errore export CSV: {e}")
    
    # 5. Verifica file creati
    print("\n5. Verifica file creati...")
    output_files = [
        ascii_output1,
        ascii_output2, 
        csv_output
    ]
    
    for file_path in output_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"   ✅ {file_path} ({file_size:.1f} KB)")
            
            # Mostra prime righe
            print(f"      Prime righe:")
            with open(file_path, 'r') as f:
                for i, line in enumerate(f):
                    if i < 5:
                        print(f"        {line.strip()}")
                    else:
                        break
            print()
        else:
            print(f"   ❌ {file_path} non creato")
    
    # 6. Informazioni WCS
    print("\n6. Informazioni WCS...")
    has_wcs = ds.attrs.get('has_wcs', 0)
    if has_wcs == 1:
        print("   ✅ WCS disponibile nel dataset")
        wcs_header = ds.attrs.get('wcs_header_dict', 'N/A')
        print(f"   WCS Header: {wcs_header[:100]}...")
    else:
        print("   ⚠️  WCS non disponibile")
        print("   I file ASCII conterranno coordinate pixel invece di coordinate geografiche")
    
    ds.close()
    print("\n=== Export Completato! ===")
    
    # 7. Istruzioni per l'uso
    print("\n📖 Come usare i file ASCII:")
    print("   - spectral_data_wcs.txt: Dati con coordinate (line, sample, lat, lon, value)")
    print("   - pixel_spectrum.txt: Spettro completo di un pixel (wavelength, value)")
    print("   - spectral_data_wcs.csv: Dati in formato CSV per Excel/analisi")

if __name__ == "__main__":
    main()













