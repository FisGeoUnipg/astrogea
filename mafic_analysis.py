import numpy as np
fron scipy.interpolation import interp1d

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def band_parameters_horgan(img_removed , wav , nbands = 5 , windows_nm = 75 , resolution_nm = 5, tol = 10):

    mappa = np.zeros((img_removed.shape[0] , img_removed.shape[1] , int(nbands*5)))
    x = wav
    for I in range(mappa.shape[0]):
        for J in range(mappa.shape[1]):

            # in this case the non-valid values are zeros, since the input img is th econtinuum removed one
            if img_removed[I,J,0] == 0:
                continue

            y = img_removed[I,J,:]

            # find the points in which the cronitnuum removed spectrum reaches the continuum ( = 1), each band is then defined as the portion
            # of the spectrum that goes below 1
            ones_indexes = np.argwhere(y == 1)
            ones_idx = []
            
            for i in range(len(ones_indexes)):
                ones_idx.append(ones_indexes[i][0])

            # iteratation used to compute the band parameters (minimum, center, depth, area and asymmetru)
            k = 0
            l = 0
            for i in range(0, len(ones_idx) - 1):
                
                band_parameters = []
        
                a = ones_idx[i+1]
                if ones_idx[i] == a:
                    continue

                
                # choose band only if the band is larger than a minimum threshold value
                if ones_idx[i+1] - ones_idx[i] >= tol:
                    k += 1
                
                    S = y[ones_idx[i]:ones_idx[i+1]]
                    X = x[ones_idx[i]:ones_idx[i+1]]
        
                    # minimum computation
                    minimum , minimum_index = np.min(S) , np.argmin(S)

                    band_parameters.append(X[minimum_index])
        
                    # center computation
                    shift_right = find_nearest(X , X[minimum_index]+windows_nm)
                    shift_left = find_nearest(X , X[minimum_index]-windows_nm)
                    Xfit_range = np.arange(X[shift_left] , X[shift_right]+resolution_nm , resolution_nm)
                    
                    interp_S = interp1d(X, S, kind='cubic' , fill_value="extrapolate")(Xfit_range)
                    
                    coeffs = np.polyfit(Xfit_range, interp_S, 4)
                    poly = np.poly1d(coeffs)
                    y_poly = poly(Xfit_range)
                    
                    idx_center = np.argmin(y_poly)
                    band_center_wav = Xfit_range[idx_center]
                    band_center_val = y_poly[idx_center]
                    
                    band_parameters.append(band_center_wav)
        
                    # depth computation
                    band_depth = 1 - S[idx_center]
                    
                    band_parameters.append(band_depth)

                    # area computation
                    total_area = np.trapz(np.ones_like(S) - S, X)
                    band_parameters.append(total_area)

                    # asymmetry computation
                    left_area = np.trapz(S[:idx_center], X[:idx_center])
                    right_area = np.trapz(S[idx_center:], X[idx_center:])
                    asymmetry = (right_area - left_area) / (100 * total_area)
                    band_parameters.append(asymmetry)
        
                    # put results in map
                    mappa[I,J,l] = X[minimum_index]
                    mappa[I,J,l+1] = Xfit_range[idx_center]
                    mappa[I,J,l+2] = band_depth
                    mappa[I,J,l+3] = total_area
                    mappa[I,J,l+4] = asymmetry

                    
                    l += 5

    return mappa
