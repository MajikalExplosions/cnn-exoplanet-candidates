from findstars import TICStar
import os
import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
import math

_mf_len = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]

_missing_frame_value = 0

def clean_stars(stars, depth = 2000, plot = False, suppress_log = False):
    for star in stars:
        clean_star(star, depth, plot = plot, suppress_log = suppress_log)

def clean_star(star, size, depth = 2000, plot = False, suppress_log = False):
    if not star.downloaded:
        return np.empty((0, 0, 0))
    data = get_star_data(star, size = size, plot = plot)
    clean = _process_nan(data)
    downscale = _downscale(clean, depth)
    normalized = _normalize(downscale, suppress_log)
    np.save(os.path.expanduser("~/.lightkurve-cache/mastDownload/TESS_processed/") + str(star.tic), normalized)
    
def get_star_data(star, size = 7, plot = False, file_path = "D:\Backup\TESS"):
    if not star.downloaded:
        print("[ERROR]", star.tic, "has not been downloaded yet.")
        return np.zeros((0, 0, 0))
    
    #Find file path and open
    ff = ""
    root_dir = os.path.expanduser(file_path)
    for directory in os.listdir(root_dir):
        if str(star.tic) in directory:
            star_dir = root_dir + "/" + directory
            ff = star_dir + "/" + os.listdir(star_dir)[0]
            break
    
    tpf = lk.TessTargetPixelFile(ff)
    if plot:
        tpf.plot(bkg = False)
    
    timestamps = tpf.time
    raw_flux = tpf.flux
    if len(timestamps) != len(raw_flux):
        print("[ERROR] Time and flux arrays have different lengths.")
        return np.zeros((0, 0, 0))
    
    #Check if we want to make 5 cuts instead of 1
    if len(raw_flux[0]) >= size + 2 and len(raw_flux[0][0]) >= size + 2:
        size += 2
    
    #Find margin sizes for both axes
    if len(raw_flux[0]) < 7 or len(raw_flux[0][0]) < 7:
        print("[ERROR] Star flux data isn't large enough")
        return np.zeros((0, 0, 0))
    margin_size_1 = (len(raw_flux[0]) - size) // 2
    margin_size_2 = (len(raw_flux[0][0]) - size) // 2
    
    #First timestamp
    arr = list()
    arr.append(raw_flux[0, margin_size_1:margin_size_1 + size, margin_size_2:margin_size_2 + size])
    
    #Make sure all time stamps are valid
    tm = tpf.nan_time_mask
    if len([i for i in range(len(tm)) if tm[i]]) != 0:
        print("[WARN] Time contains nan.")
        
    #Second timestamp to end; we do this because if there are missing frames between A and B *before* B.
    for i in range(1, len(timestamps)):
        #Get the time between this frame and the previous one; we minimize the total error for frame count vs minutes
        #By adding frames until adding more frames would increase error (which can be done by rounding).
        delta_minutes = (timestamps[i].value - timestamps[i - 1].value) * 1440
        missing_frames = round(delta_minutes / 2 - 1)
        #Add data to _mf_len for plotting later
        if missing_frames != 0:
            _mf_len[1 if star.is_planet_candidate else 0][math.ceil(math.log10(missing_frames))] += 1
        
        #Add the correct number of missing frames; we round to the nearest whole number.
        while missing_frames >= 1:
            missing_frames -= 1
            arr.append(np.full((size, size), _missing_frame_value, dtype=np.float64))
        #Add the actual frame itself
        arr.append(raw_flux[i][margin_size_1:margin_size_1 + size, margin_size_2:margin_size_2 + size])
    return np.array(arr)

def _process_nan(arr):
    side_len = len(arr[0])
    processed = list()
    for row in arr:
        #If there are no nans, add it to the final. If there are, we add empty array.
        if not np.isnan(row).any():
            processed.append(row)
        else:
            processed.append(np.full((side_len, side_len), _missing_frame_value, dtype=np.float64))
    final = np.array(processed)
    
    if np.isnan(final).any():
        print("[WARN] nan could not be removed.")
        
    return final

def _downscale(arr, new_length):
    arr = arr.copy()
    averaged = []
    #Sections aren't distributed evenly; should be okay regardless, though
    #Split array into new_length subarrays and iterate
    for subarray in np.array_split(arr, new_length):
        #Convert all 0's to nan
        subarray[subarray == 0] = np.nan
        #nanmean excludes all nan's, so all former 0's are excluded from the mean. Then convert the nan's back to 0's using nan_to_num.
        averaged.append(np.nan_to_num(np.nanmean(subarray, axis=0)))
    
    return np.array(averaged)
    

def _normalize(arr, suppress_log):
    arr = np.copy(arr)
    
    #Find the highest median value for all pixels
    sl1, sl2 = len(arr[0]) // 2, len(arr[0][0]) // 2
    bm = _get_median_brightest(arr)
    removed_frames = 0
    for i in range(len(arr)):
        #Any data that is more than 1.5x compared to the median of the brightest we count as invalid.
        if np.max(arr[i]) > bm * 1.5:
            arr[i] = np.full(arr[i].shape, _missing_frame_value, dtype=np.float64)
            removed_frames += 1
        #Any data that's significantly negative we also assume is bad.
        elif np.min(arr[i]) < bm * -0.5:
            arr[i] = np.full(arr[i].shape, _missing_frame_value, dtype=np.float64)
            removed_frames += 1
    
    #Normalize data by setting 1 to the new median of the brightest pixel
    arr /= _get_median_brightest(arr)
    if removed_frames > len(arr) // 2:
        print("[WARN] Too many removed frames (" + str(removed_frames) + "); remove star.")
    elif not suppress_log:
        print("Removed", removed_frames, "frames.")
    return arr, removed_frames

def _get_median_brightest(arr):
    medians = np.median(arr, axis=0)
    return np.max(medians)

def _plot_approximate_lightcurve(arr, title, draw_line = True, strict_bounds = False, hdpi = False, line_y = 1):
    
    if np.isnan(arr).any():
        print("[WARN] Contains nan.")
        arr = np.nan_to_num(arr)
    
    fig = plt.figure(dpi=(160 if hdpi else 80), figsize=(64, 4))
    ax = fig.add_subplot(1, 1, 1)
    x, y = [], []
    count = 0
    sl1, sl2 = len(arr[0]) // 2, len(arr[0][0]) // 2
    for frame in arr:
        x.append(count)
        count += 1
        mx = np.max(frame[sl1 - 1:sl1 + 2, sl2 - 1:sl2 + 2])
        y.append(mx)
    ax.scatter(x, y, s=1)
    if draw_line:
        ax.plot([0, len(arr)], [line_y, line_y], ls="--", c="r", alpha=0.5)
    ax.set_title(title)
    ax.set_xlim(0, len(arr))
    if strict_bounds:
        med = np.median(y)
        ax.set_ylim(-med * 0.1, med * 1.2)
    
    plt.show()