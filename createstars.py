import random, math
import numpy as np
from cleanstars import _normalize
import os

_bad_frame_probs = []
_dataset_arr = []

def create_stars(count, size = 7, cut_5 = True, index = 1, read_bad = True, save = True):
    if cut_5:
        size += 2
    np_size = (2000, size, size)
    output = []
    
    ind = 0
    while ind < count:
        try:
            img = np.empty(np_size)
            si = _StarImage(size, read_bad)
            for i in range(np_size[0]):
                for j in range(np_size[1]):
                    for k in range(np_size[2]):
                        img[i][j][k] = si.get_pixel(j, k, i)
            norm, rf = _normalize(img, True)
        except OverflowError:
            #If we have the funny cosh error thing then pretend we never generated this star.
            continue
            
        if rf == 2000:
            continue
        #Add to arr and save and return only if it's valid star
        
        output.append((si.base_star.has_exoplanet, norm))
        if save:
            n = len(_dataset_arr)
            np.save(os.path.expanduser("~/.lightkurve-cache/mastDownload/TESS_processed/") + str(index * 10000 + n), output[-1][1])
            _dataset_arr.append([index * 10000 + n, 1 if si.base_star.has_exoplanet else 0])
        ind += 1
    
    return output

def save_stars():
    np.savetxt("res/dataset_sim.csv", np.array(_dataset_arr, dtype=np.int32))

class _StarImage:
    def __init__(self, size, read_bad):
        self.np_size = (2000, size, size)
        max_rad = size / 3
        #Base star (center)
        self.base_star = _Star(_rnorm(0.5, max_rad))
        
        #Add side stars
        self.side_stars = []
        
        #Larger stars
        count = 0
        while random.random() > 0.4 and count < 8:
            #Find position of new star
            position = [random.randint(2, size // 2 + 1), random.randint(2, size // 2 + 1)]
            if random.random() > 0.5:
                position[0] *= -1
            if random.random() > 0.5:
                position[1] *= -1
            position = [position[0] + size // 2, position[1] + size // 2]
            star_size = _rnorm(0.1, max_rad)
            brightness = _rrange(0.1, 1.9)
            self.side_stars.append((position, brightness, _Star(star_size)))
            count += 1
        
        #Smaller stars
        count = 0
        while random.random() > 0.8 and count < 24:
            #Find position of new star
            position = [random.randint(2, size // 2 + 1), random.randint(2, size // 2 + 1)]
            if random.random() > 0.5:
                position[0] *= -1
            if random.random() > 0.5:
                position[1] *= -1
            position = [position[0] + size // 2, position[1] + size // 2]
            star_size = _rnorm(0.1, max_rad ** 2)
            brightness = _rrange(0.01, 0.6) ** 3
            self.side_stars.append((position, brightness, _Star(star_size)))
            count += 1
        
        #Bad filters
        if read_bad:
            _read_bad_frames()
            self._bad_filter = []
            num_bad = round(max(0, _rnorm(0, 6.5 * 2)))
            for i in range(num_bad):
                d = random.randrange(0, len(_bad_frame_probs))
                while d in self._bad_filter:
                    d = random.randrange(0, len(_bad_frame_probs))
                self._bad_filter.append(d)
        else:
            self._bad_filter = []
        
        #Cache values
        self._background = _rnorm(-0.02, 0.03)
        self.center = size // 2
    
    def get_pixel(self, x, y, t):
        pixel = self._get_layer_background()
        pixel += self._get_layer_noise()
        pixel += self._get_layer_center(x, y, t)
        pixel += self._get_layer_side(x, y, t)
        if self._get_layer_bad(t) == -1:
            pixel = -100 * abs(pixel)
        return pixel
        
    def _get_layer_noise(self, mag = 0.005):
        return _rnorm(-1 * mag, mag)
    
    def _get_layer_background(self):
        return self._background
    
    def _get_layer_center(self, x, y, t):
        return self.base_star.get_pixel_value(x - self.center, y - self.center, t)
    
    def _get_layer_side(self, x, y, t):
        res = 0
        for star in self.side_stars:
            res += star[2].get_pixel_value(x - star[0][0], y - star[0][1], t) * star[1]
        return res
    
    def _get_layer_bad(self, t):
        for b in self._bad_filter:
            if _bad_frame_probs[b][0] <= t and _bad_frame_probs[b][1] >= t:
                return -1
        return 0

class _Star:
    def __init__(self, size):
        self.size = size
        self.mods = []
        self.has_exoplanet = False
        self.offset = (_rrange(-0.5, 0.5), _rrange(-0.5, 0.5))
        if random.random() < 0.1:
            self.has_exoplanet = True
            #Add pulse modification
            #(Mod Type 1, Strength, Period, Offset)
            self.mods.append([1, _rnorm(-0.02, 0.02), _rrange(30 * 12, 4000), _rrange(0, 2000)])
            self.mods[-1][3] = self.mods[-1][3] % self.mods[-1][2]
            self.mods[-1][1] = abs(self.mods[-1][1])
            
        if random.random() < 0.5:
            self.has_exoplanet = True
            #Add exoplanet transit modification
            #(Mod Type 2, Strength, Period, Offset, Length)
            self.mods.append([2, _rnorm(0, 0.02), _rnorm(30 * 12, 2000), _rrange(0, 2000), _rnorm(5, 25)])
            self.mods[-1][3] = self.mods[-1][3] % self.mods[-1][2]
            self.mods[-1][1] = max(self.mods[-1][1], 0)
            
        if random.random() < 0.15:
            #(Mod Type 3, Strength, Period, Offset)
            #Add jump modification
            self.mods.append([3, _rrange(0, 0.1), _rrange(30 * 12, 4000), _rrange(0, 2000)])
            self.mods[-1][1] = max(self.mods[-1][1], 0)
        
    
    def get_pixel_value(self, x, y, frame_num):
        #base_value = -1 * ((x - self.offset[0]) ** 2 + (y - self.offset[1]) ** 2) ** 4 / self.size + 1 #(x^2+y^2)^2, to make the bottom "flatter"
        #base_value = 0.5 * (math.cos(math.pi * _dist(x - self.offset[0], y - self.offset[1], 0) / self.size) + 1)
        d = _dist(x - self.offset[0], y - self.offset[1], 0)
        base_value = 1 / math.cosh(math.pi * d / self.size)
        for mod in self.mods:
            if mod[0] == 1:
                #Pulse
                mod_step = (frame_num - mod[3]) % mod[2]
                base_value += mod[1] * math.sin(mod_step / mod[2] * 2 * math.pi)
            if mod[0] == 2:
                #Transit
                #TODO make transit a single line rather than general dimming
                mod_step = (frame_num - mod[3]) % mod[2]
                if mod_step < mod[4]:
                    #Dip the graph
                    base_value += mod[1] * (math.cos(mod_step / mod[4] * 2 * math.pi) - 1) / 2
            if mod[0] == 3:
                #Jump
                mod_step = (frame_num - mod[3]) % mod[2]
                base_value += mod[1] * mod_step / mod[2]
        
        return max(base_value, 0)
       
def _rnorm(x, y):
    m = (x + y) / 2
    return np.random.normal(m, (y - m) / 3)

def _rrange(x, y):
    return random.random() * (y - x) + x

def _dist(x, y, center):
    return math.sqrt(((x - center) ** 2 + (y - center) ** 2))

def _read_bad_frames():
    if len(_bad_frame_probs) != 0:
        return
    #Load data
    f = open("res/dataset.csv")
    raw = f.readlines()
    f.close()
    missing = []
    for line in raw:
        line = [int(s.strip()) for s in line.split()]
        data = np.load(os.path.expanduser("~/.lightkurve-cache/mastDownload/TESS_processed/") + str(line[0]) + ".npy")
        center1 = len(data[0]) // 2
        center2 = len(data[0][0]) // 2
        last_was_missing = False
        first_frame = -1
        for frame in range(len(data)):
            is_bad = data[frame][center1][center2] <= 0
            if is_bad:
                #If going from good to bad
                if not last_was_missing:
                    last_was_missing = True
                    first_frame = frame
            else:
                #Going from bad to good
                if last_was_missing:
                    missing.append(first_frame * 10000 + frame)
                    last_was_missing = False
    
    missing.sort()
    for item in missing:
        _bad_frame_probs.append((item // 10000, item % 10000))
    