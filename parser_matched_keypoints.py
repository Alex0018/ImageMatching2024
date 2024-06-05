import h5py
import os
from pathlib import Path
import numpy as np



class ParserMatchedKeypoints:
    MULT = 10000000

    def __init__(self, 
                 threshold_low:  float = 0.3,
                 filename_keypoints='keypoints.h5', 
                 filename_matches='matches.h5'):
        self.filename_keypoints = filename_keypoints
        self.filename_matches = filename_matches
        self.threshold_low = threshold_low


    def run(self, dir_images:     Path, 
                  dir_features:   Path, 
                  filename:       Path,
                  ):
        keypoints = self._get_keypoints(dir_images, filename)
        self._save_keypoints(keypoints, dir_features)

        matches = self._get_matches(filename, keypoints)
        self._save_matches(matches, dir_features)

    
    def title(self):
        return f'matchesT{self.threshold_low}'
    
    def get_params_dict(self):
        return {'threshold_low': self.threshold_low}


    def _get_keypoints(self, dir_images, filename):
        keypoints = {fname: [] for fname in os.listdir(dir_images) if fname.endswith('.png')}

        with h5py.File(filename, mode='r') as f_match:
            for key1 in f_match.keys():        
                for key2 in f_match[key1].keys():
                    coords = f_match[key1][key2]
                    for (x1, y1, x2, y2, _) in coords:
                        keypoints[key1].append(int(x1) * ParserMatchedKeypoints.MULT + int(y1))
                        keypoints[key2].append(int(x2) * ParserMatchedKeypoints.MULT + int(y2))

        # make a lookup dictionary
        empty_keys = []
        for key in keypoints:
            if len(keypoints[key]) > 0:
                tmp = sorted(list(set(keypoints[key])))
                keypoints[key] = {tmp[i]: i for i in range(len(tmp))}
            else:
                empty_keys.append(key)        
        for key in empty_keys:
            del keypoints[key]

        return keypoints


    def _save_keypoints(self, keypoints, dir_features):        
        with h5py.File(dir_features / self.filename_keypoints, mode='w') as file_keypoints:
            for key in keypoints:
                file_keypoints[key] = np.array([(k // ParserMatchedKeypoints.MULT, k % ParserMatchedKeypoints.MULT) for k in keypoints[key]])


    def _get_matches(self, filename, keypoints):
        matches = {}
        keys_without_matches = []
        with h5py.File(filename, mode='r') as f_match:
            for key1 in f_match.keys():    
                matches[key1] = {}    
                for key2 in f_match[key1].keys():
                    coords = f_match[key1][key2]

                    # leave only coordinates with confidence higher than threshold_low
                    coords = [c for c in coords if c[4] > self.threshold_low] 

                    if len(coords) == 0:
                        continue

                    matches[key1][key2] = np.zeros((len(coords), 2), dtype=int)

                    for i, (x1, y1, x2, y2, _) in enumerate(coords):
                        matches[key1][key2][i] = keypoints[key1][int(x1) * ParserMatchedKeypoints.MULT + int(y1)], \
                                                 keypoints[key2][int(x2) * ParserMatchedKeypoints.MULT + int(y2)]
                        
                if len(matches[key1]) == 0:
                    keys_without_matches.append(key1)

        for key in keys_without_matches:
            del matches[key]

        return matches


    def _save_matches(self, matches, dir_features):
        with h5py.File(dir_features / self.filename_matches, mode='w') as f_matches:
            for key1 in matches:
                group  = f_matches.require_group(key1)
                for key2 in matches[key1]:
                    group.create_dataset(key2, data=matches[key1][key2])