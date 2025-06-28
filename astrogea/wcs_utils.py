import numpy as np
from typing import List, Optional
import warnings

def parse_envi_map_info_list(map_info_list: List[str]) -> Optional[dict]:
    """
    Parse ENVI map info list to extract georeferencing information.
    Matches the functionality of the reference script.
    """
    if not isinstance(map_info_list, list) or len(map_info_list) < 8:
        return None
    
    try:
        parts = [str(p).strip() for p in map_info_list]
        info = {
            'projection_name': parts[0],
            'ref_x_1based': float(parts[1]),
            'ref_y_1based': float(parts[2]),
            'ref_coord1': float(parts[3]),
            'ref_coord2': float(parts[4]),
            'pixel_size_x': float(parts[5]),
            'pixel_size_y': float(parts[6]),
            'units': 'unknown',
            'datum': 'unknown'
        }
        
        # Parse units from the remaining parts
        for part in reversed(parts[7:]):
            part_lower = part.lower()
            if 'units=' in part_lower:
                unit_val = part_lower.split('units=')[-1].strip()
                if 'meter' in unit_val:
                    info['units'] = 'm'
                elif 'deg' in unit_val:
                    info['units'] = 'deg'
                else:
                    info['units'] = unit_val
                break
            elif 'meter' in part_lower:
                info['units'] = 'm'
                break
            elif 'deg' in part_lower:
                info['units'] = 'deg'
                break
        
        # Parse datum
        potential_datum_idx = -2 if 'units=' in parts[-1].lower() else 7
        if len(parts) > potential_datum_idx:
            try:
                float(parts[potential_datum_idx])
            except ValueError:
                info['datum'] = parts[potential_datum_idx]
        
        # Set coordinate names based on units
        if info['units'] == 'm':
            info['ref_easting'], info['ref_northing'] = info['ref_coord1'], info['ref_coord2']
        elif info['units'] == 'deg':
            info['ref_lon'], info['ref_lat'] = info['ref_coord1'], info['ref_coord2']
        
        return info
    except (ValueError, IndexError, KeyError) as e:
        warnings.warn(f"Error parsing 'map info' list: {e}")
        return None

def create_wcs_from_parsed_info(parsed_info: dict, shape: tuple) -> Optional['WCS']:
    """
    Create astropy WCS object from parsed map info.
    Matches the functionality of the reference script.
    """
    if parsed_info is None:
        return None
    
    try:
        from astropy.wcs import WCS
        
        lines, samples = shape
        units = parsed_info.get('units', 'unknown')
        proj_name = parsed_info.get('projection_name', '').lower()
        
        w = WCS(naxis=2)
        w.wcs.crpix = [parsed_info['ref_x_1based'], parsed_info['ref_y_1based']]
        w.wcs.cdelt = [parsed_info['pixel_size_x'], -abs(parsed_info['pixel_size_y'])]
        
        ctype1, ctype2 = '', ''
        units_lower = units.lower()
        
        if units_lower == 'm' or units_lower == 'meters':
            w.wcs.crval = [parsed_info['ref_easting'], parsed_info['ref_northing']]
            w.wcs.cunit = ['m', 'm']
            ctype1, ctype2 = 'XMETR', 'YMETR'
        elif units_lower == 'deg' or units_lower == 'degree' or units_lower == 'degrees':
            w.wcs.crval = [parsed_info['ref_lon'], parsed_info['ref_lat']]
            w.wcs.cunit = ['deg', 'deg']
            ctype1, ctype2 = 'OLON', 'OLAT'
            
            # Add projection suffixes
            if 'sinusoidal' in proj_name:
                ctype1 += '-SIN'
                ctype2 += '-SIN'
            elif 'equirectangular' in proj_name or 'plate carree' in proj_name:
                ctype1 += '-CAR'
                ctype2 += '-CAR'
            elif 'lambert conformal' in proj_name:
                ctype1 += '-LCC'
                ctype2 += '-LCC'
            elif 'polar stereographic' in proj_name:
                ctype1 += '-STG'
                ctype2 += '-STG'
        else:
            warnings.warn(f"Units '{units}' not handled.")
            return None
        
        w.wcs.ctype = [ctype1, ctype2]
        return w
        
    except ImportError:
        warnings.warn("astropy.wcs not available, cannot create WCS object")
        return None
    except (KeyError, Exception) as e:
        warnings.warn(f"WCS creation failed - {type(e).__name__}: {e}")
        return None

def create_wcs_header_dict(wcs_object: 'WCS', parsed_info: dict) -> Optional[dict]:
    """
    Create WCS header dictionary for NetCDF attributes.
    """
    if wcs_object is None:
        return None
    
    try:
        wcs_header = wcs_object.to_header(relax=True)
        wcs_header_dict = dict(wcs_header)
        wcs_header_dict = {k: v for k, v in wcs_header_dict.items() if v is not None}
        
        # Add additional metadata
        if 'projection_name' in parsed_info:
            wcs_header_dict['PROJNAME'] = parsed_info['projection_name']
        if 'datum' in parsed_info:
            wcs_header_dict['DATUM'] = parsed_info['datum']
        
        return wcs_header_dict
    except Exception as e:
        warnings.warn(f"Error creating WCS header: {e}")
        return None
