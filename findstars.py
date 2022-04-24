import pandas
import astroquery.mast
import csv
import lightkurve as lk
import os
import shutil
import random
from astropy import coordinates

class TICStar:
    _stars = dict()
    _toi_catalog = None
    
    def __init__(self, tic, sync = True):
        #The TIC of the star
        self.tic = int(tic)
        #Whether the star's TPF is downloaded
        self.downloaded = False
        #A table of the star's TPFs
        self.tpfs = None
        #Whether the star is valid or not; 0 is no, 1 is yes. -1 is unknown.
        self._valid = -1
        #Whether the star is a planet candidate or not
        self.is_planet_candidate = self.tic in TICStar._toi_catalog["TIC"].values
        
        TICStar._stars[tic] = self
        if sync:
            TICStar._sync_star_cache()
    
    """Finds all TPFs for this star and stores them"""
    def find_tpfs(self):
        
        if self.tpfs is None:
            self.tpfs = lk.search_targetpixelfile("TIC " + str(self.tic), cadence="short", mission="TESS")
        
        if self.is_planet_candidate:
            if len(self.tpfs) != 1:
                #Planet candidates have only 1 valid TPF
                #Check TOI Catalog
                toi_row = TICStar._toi_catalog.loc[TICStar._toi_catalog["TIC"] == self.tic]
                toi_sectors = [int(i) for i in str(toi_row["Sectors"].iloc[0]).strip().split()]

                missions = self.tpfs
                mission = [m for m in missions if int(m.table["mission"][0].split()[-1].strip()) == toi_sectors[0]]
                
                self.tpfs = mission[0]
                if len(mission) != 1:
                    print("[WARN] More than 1 TPF for target sector")
    """Downloads the first TPF for this star. We only want one per star to make sure no star is weighed more than any other."""
    def download_tpfs(self):
        if self.downloaded:
            return
        
        if not self.is_valid_star():
            return
        
        self.find_tpfs()
        self.tpfs.download()
        self.downloaded = True
        TICStar._sync_star_cache()
    
    """Checks if this star is valid. A valid star either: 1) shows up in 1 matching sector in both TPF and TOI
    or 2) shows up in at least one TPF sector and none in TOI. While only one is downloaded, having more is okay."""
    def is_valid_star(self, strict = False):
        #Valid checks if we need to check for TPFs
        if self._valid == 0:
            return False
        if self._valid == 1:
            return True
        
        #Check TOI Catalog
        toi_row = TICStar._toi_catalog.loc[TICStar._toi_catalog["TIC"] == self.tic]
        #Check if it appears in TOI catalog; if yes, get the list of sectors
        if len(toi_row) != 0:
            toi_sectors = [int(i) for i in str(toi_row["Sectors"].iloc[0]).strip().split()]
        else:
            toi_sectors = list()
        
        #Make sure it appears in 0 or 1 sectors; if not, it's invalid anyways.
        if len(toi_sectors) > 1:
            self._valid = 0
            TICStar._sync_star_cache()
            return False
        
        #Check TPFs for TOI stars
        self.find_tpfs()
        if len(toi_sectors) == 1:
            if strict:
                if len(self.tpfs.table) != 1:
                    self._valid = 0
                    TICStar._sync_star_cache()
                    return False
                else:
                    #mission is something like "TESS Sector 1"
                    mission = self.tpfs.table[0]["mission"]
                    tpf_sector = int(mission.split()[-1].strip())

                    self._valid = 1 if toi_sectors[0] == tpf_sector else 0
                    TICStar._sync_star_cache()
                    return toi_sectors[0] == tpf_sector
            else:
                if len(self.tpfs.table) == 0:
                    self._valid = 0
                    TICStar._sync_star_cache()
                    return False
            
                if len(toi_sectors) == 1:
                    missions = self.tpfs.table["mission"]
                    missions = [int(mission.split()[-1].strip()) for mission in list(missions)]
                    
                    self._valid = 1 if toi_sectors[0] in missions else 0
                    TICStar._sync_star_cache()
                    return toi_sectors[0] in missions
                
        #Star only shows up in TPF sectors, so it's valid.
        self._valid = 1
        TICStar._sync_star_cache()
        return True
    
    """Finds random stars that aren't in the TOI Catalog and returns them. Returns a list of TICStar with random length."""
    def find_random_stars(search_radius = 600):
        coord = coordinates.SkyCoord(TICStar._rand_range(0, 360), TICStar._rand_range(-90, 90), unit="deg")
        res_tpfs = lk.search_targetpixelfile(coord, cadence="short", mission="TESS", radius=search_radius)
        res = []
        
        for tpf in res_tpfs:
            toi = int(tpf.target_name[0])
            #Add star to result if it isn't in TOI catalog and isn't already created
            if toi not in TICStar._toi_catalog["TIC"].values and toi not in TICStar._stars.keys():
                res.append(TICStar(toi))
                break
        
        return res
    
    """Reads the TOI Catalog, syncs _stars with the cache, and cleans downloads"""
    def load(clean = True):
        
        toi_file = open("res/toi_catalog.csv", "r", encoding="UTF-8")
        csv_reader = csv.reader(toi_file, delimiter=',', quotechar='"')
        csv_data = list()
        for row in csv_reader:
            csv_data.append(row)
            for i in range(len(csv_data[-1])):
                csv_data[-1][i] = csv_data[-1][i].strip()
                try:
                    csv_data[-1][i] = float(csv_data[-1][i])
                    if round(csv_data[-1][i]) == csv_data[-1][i]:
                        csv_data[-1][i] = int(csv_data[-1][i])
                except:
                    pass
                
        TICStar._toi_catalog = pandas.DataFrame(csv_data[5:], columns=csv_data[4])
        
        #Add all stars in TOI Catalog
        for i, row in TICStar._toi_catalog.iterrows():
            star = TICStar(row["TIC"], sync = False)

        TICStar._sync_star_cache()
        
        if clean:
            TICStar._clean_downloads()
        else:
            TICStar._add_existing_downloads()
        
    
    """Removes the "TIC" prefix if one exists and converts the result to an integer. Throws an error if 
    the conversion was unsuccessful"""
    def _remove_tic_prefix(tic):
        if isinstance(tic, int):
            return tic
        if tic[:4] == "TIC ":
            tic = tic[4:]
        elif tic[:3] == "TIC":
            tic = tic[3:]
        
        #Return the integer portion; if it isn't valid, it should throw an error
        return int(tic)
    
    """Syncs the cache file with the stored list."""
    def _sync_star_cache():
        #Read items in cache
        cache = []
        cache_path = os.path.expanduser("~/.lightkurve-cache/mastDownload/_tess_cache.txt")
        if os.path.isfile(cache_path):
            f = open(cache_path, "r")
            cache = f.readlines()
            f.close()
        
        #Add items in cache to _downloaded
        for row in cache:
            if len(row) == 0:
                continue
            #This row has stuff, so assume it's a valid row.
            row = row.strip().split()
            row = [int(i.strip()) for i in row]
            tic, downloaded, valid = row[0], row[1] == 1, row[2]
            
            if tic in TICStar._stars.keys():
                #Mark downloaded as true if it's downloaded in either cache or object
                TICStar._stars[tic].downloaded = TICStar._stars[tic].downloaded or downloaded

                #Only update validity if cache has a value
                if valid != -1:
                    #If the object has no value, update with the cache's value
                    if TICStar._stars[tic]._valid == -1:
                        TICStar._stars[tic]._valid = valid
                    #If the object has a value and it's different, then we have an error. Otherwise object is up-to-date.
                    elif valid != TICStar._stars[tic]._valid:
                        print("[ERROR]", tic, "validity desynced.")
            else:
                #Doesn't exist, so create new.
                TICStar._stars[tic] = TICStar(tic, sync = False)
                TICStar._stars[tic].downloaded = downloaded
                TICStar._stars[tic]._valid = valid
        
        #Write the new combined _downloaded back to the cache
        output = ""
        for tic in TICStar._stars.keys():
            s = TICStar._stars[tic]
            output += str(s.tic) + " " + str(1 if s.downloaded else 0) + " " + str(s._valid)
            output += "\n"
        
        f = open(cache_path, "w")
        f.truncate()
        f.write(output[:-1])
        f.close()
    
    """Cleans the downloads folder by deleting partial downloads."""
    def _clean_downloads():
        TICStar._sync_star_cache()
        root_folder = os.path.expanduser("~/.lightkurve-cache/mastDownload/TESS")
        if not os.path.isdir(root_folder):
            return
        
        download_folders = os.listdir(root_folder)
        for i in range(len(download_folders)):
            tic = int(download_folders[i].split("-")[2].lstrip('0'))
            if tic not in TICStar._downloaded_stars():
                print("Removing", tic)
                shutil.rmtree(root_folder + "/" + download_folders[i])
    
    """Returns a list of the (integer) TICs of all downloaded stars"""
    def _downloaded_stars():
        return [key for key in TICStar._stars.keys() if TICStar._stars[key].downloaded]
    
    """Returns a random float in range [a, b)."""
    def _rand_range(a, b):
        return random.random() * (b - a) + a
    
    def _add_existing_downloads():
        TICStar._sync_star_cache()
        root_folder = os.path.expanduser("~/.lightkurve-cache/mastDownload/TESS")
        if not os.path.isdir(root_folder):
            return
        
        download_folders = os.listdir(root_folder)
        for i in range(len(download_folders)):
            tic = int(download_folders[i].split("-")[2].lstrip('0'))
            if tic in TICStar._stars.keys():
                #print("Adding", tic)
                TICStar._stars[tic].downloaded = True
            else:
                #print("Found", tic, "but no such star exists. Creating.")
                #Doesn't exist, so create new.
                TICStar._stars[tic] = TICStar(tic, sync = False)
                TICStar._stars[tic].downloaded = True
        
        TICStar._sync_star_cache()
