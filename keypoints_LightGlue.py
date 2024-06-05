# Reference: 
# https://www.kaggle.com/code/motono0223/imc-2024-multi-models-pipeline/notebook


import numpy as np

import kornia as K
import kornia.feature as KF
from lightglue import ALIKED, SuperPoint, DoGHardNet, DISK, SIFT
import torch

from typing import List, Tuple

from tqdm import tqdm
import h5py
from pathlib import Path
import cv2

import gc

from src.keypoints_matcher_base import KeypointsMatcherBase



class Keypoints_LightGlue(KeypointsMatcherBase):

    FILENAME_KEYPOINTS = 'keypoints_LightGlue.h5'
    FILENAME_KEYPOINTS_DESCRIPTORS = 'descriptors_LightGlue.h5'
    FILENAME_KEYPOINTS_FAST = 'keypoints_fast.h5'
    FILENAME_KEYPOINTS_DESCRIPTORS_FAST = 'descriptors_fast.h5'

    dict_extractors = { 'aliked':      ALIKED,
                        'superpoint':  SuperPoint,
                        'doghardnet':  DoGHardNet,
                        'disk':        DISK,
                        'sift':        SIFT}

    def __init__(self, 
                 extractor_class_label='aliked',
                 resize_to=1024,
                 device=torch.device('cpu')):
        super(Keypoints_LightGlue, self).__init__()
        
        self.device = device

        self.extractor_class_label = extractor_class_label
        self.resize_to = resize_to

        self.extractor_dtype = torch.float32 # ALIKED has issues with float16
        extractor_class = Keypoints_LightGlue.dict_extractors.get(extractor_class_label)

        self.extractor = extractor_class(
                max_num_keypoints=4096, 
                detection_threshold=0.01, 
                resize=resize_to
            ).eval().to(device, self.extractor_dtype)
        self.extractor_fast = extractor_class(
                max_num_keypoints=1024, 
                detection_threshold=0.001, 
                resize=256
            ).eval().to(device, self.extractor_dtype)

        matcher_params = {
            'width_confidence': -1,
            'depth_confidence': -1,
            'mp': True if 'cuda' in str(device) else False,
        }
        self.matcher = KF.LightGlueMatcher(extractor_class_label, matcher_params).eval().to(device)



    def title(self):
        return f'{self.extractor_class_label}{self.resize_to}_LightGlue'
    

    def get_params_dict(self):
        return {'extractor_class_label': self.extractor_class_label,
                'resize_to': self.resize_to}


    def detect_keypoints(self, image_paths, dir_features):
        keypoints_dict = {}
        descriptors_dict = {}
        keypoints_small_dict = {}
        descriptors_small_dict = {}

        for path in tqdm(image_paths, desc='Computing keypoints'):             
            with torch.inference_mode():               
                image = self._load_torch_image(path).to(self.extractor_dtype)

                key = path.name
                keypoints_dict[key] = {}
                descriptors_dict[key] = {}
                for rotation in range(4):
                    # torch.cuda.empty_cache()
                    img = image.rot90(rotation, dims=[2,3]).to(device=self.device)
                    features = self.extractor.extract(img)
                    keypoints_dict[key][rotation]   = features['keypoints'  ].squeeze().detach().cpu().numpy()
                    descriptors_dict[key][rotation] = features['descriptors'].squeeze().detach().cpu().numpy()

                keypoints_small_dict[key] = {}
                descriptors_small_dict[key] = {}
                for rotation in range(4):
                    img = image.rot90(rotation, dims=[2,3]).to(device=self.device)
                    features = self.extractor_fast.extract(img)
                    keypoints_small_dict[key][rotation]   = features['keypoints'  ].squeeze().detach().cpu().numpy()
                    descriptors_small_dict[key][rotation] = features['descriptors'].squeeze().detach().cpu().numpy()    


        self._save_to_h5(dir_features / Keypoints_LightGlue.FILENAME_KEYPOINTS, keypoints_dict)
        self._save_to_h5(dir_features / Keypoints_LightGlue.FILENAME_KEYPOINTS_DESCRIPTORS, descriptors_dict)
        self._save_to_h5(dir_features / Keypoints_LightGlue.FILENAME_KEYPOINTS_FAST, keypoints_small_dict)
        self._save_to_h5(dir_features / Keypoints_LightGlue.FILENAME_KEYPOINTS_DESCRIPTORS_FAST, descriptors_small_dict)
        
        return keypoints_dict, descriptors_dict, keypoints_small_dict, descriptors_small_dict




    def run(self, image_paths, index_pairs, dir_features, filename_save, min_matches=15):

        count_matched_pairs = 0

        if not (dir_features / Keypoints_LightGlue.FILENAME_KEYPOINTS).exists():
            self.detect_keypoints(image_paths, dir_features)

        keypoints_dict = self._load_h5_to_dict(dir_features / Keypoints_LightGlue.FILENAME_KEYPOINTS)
        descriptors_dict = self._load_h5_to_dict(dir_features / Keypoints_LightGlue.FILENAME_KEYPOINTS_DESCRIPTORS)
        keypoints_small_dict = self._load_h5_to_dict(dir_features / Keypoints_LightGlue.FILENAME_KEYPOINTS_FAST)
        descriptors_small_dict = self._load_h5_to_dict(dir_features / Keypoints_LightGlue.FILENAME_KEYPOINTS_DESCRIPTORS_FAST)

        image_shapes = [cv2.imread(str(path)).shape[:2] for path in image_paths]
        matches = {path.name:{} for path in image_paths}
           
        for (idx1, idx2, _) in tqdm(index_pairs, desc='Matching keypoints'):

            key1, key2 = image_paths[idx1].name, image_paths[idx2].name
            image2_original_shape = image_shapes[idx2] 

            keypoints1, keypoints2, confidences = self._match_image_pair_with_rotation(
                                                            keypoints_dict, descriptors_dict, 
                                                            keypoints_small_dict,  descriptors_small_dict, 
                                                            key1, key2, 
                                                            image2_original_shape)

            if len(confidences) >= min_matches:
                matches[key1][key2] = np.concatenate([keypoints1, keypoints2, confidences], axis=1).astype(np.float32)
                count_matched_pairs += 1

        self._save_to_h5(dir_features / filename_save, matches)
                
        return count_matched_pairs


    def _save_to_h5(self, path_save, dict):
        with h5py.File(path_save, mode='w') as f_save: 
            for key1 in dict:
                group  = f_save.require_group(key1)
                for key2 in dict[key1]:
                    group.create_dataset(str(key2), data=dict[key1][key2])

    def _load_h5_to_dict(self, path):
        dict = {}
        with h5py.File(path, mode='r') as f_load: 
            for key1 in f_load.keys():
                dict[key1] = {}
                for key2 in f_load[key1].keys():
                    dict[key1][key2] = f_load[key1][key2][:]
        return dict


    def _match_image_pair_with_rotation(self, 
                                        keypoints_dict,        descriptors_dict, 
                                        keypoints_small_dict,  descriptors_small_dict, 
                                        key1,                  key2, 
                                        image2_original_shape):
        
        rot2 = self._get_best_rotation(keypoints_small_dict[key1], 
                                       keypoints_small_dict[key2], 
                                       descriptors_small_dict[key1], 
                                       descriptors_small_dict[key2], )   
        keypoints1 = keypoints_dict[key1]['0']
        keypoints2 = keypoints_dict[key2][str(rot2)]
        indice_matches, distances = self._match_image_pair( torch.from_numpy(keypoints1).to(self.device), 
                                                            torch.from_numpy(keypoints2).to(self.device), 
                                                            torch.from_numpy(descriptors_dict[key1]['0']).to(self.device), 
                                                            torch.from_numpy(descriptors_dict[key2][str(rot2)]).to(self.device) )    

        keypoints_coordinates_1 = self._get_coordinates(keypoints1, indice_matches, 0)
        keypoints_coordinates_2 = self._get_coordinates(keypoints2, indice_matches, 1)

        if  (len(keypoints_coordinates_1) > 0) and \
            (len(keypoints_coordinates_2) > 0) and \
            (keypoints_coordinates_1.shape[1] == 2) and \
            (keypoints_coordinates_2.shape[1] == 2):
            return (keypoints_coordinates_1, 
                    KeypointsMatcherBase._rotate_keypoints(keypoints_coordinates_2, *image2_original_shape, rot2),
                    distances.reshape(-1, 1).detach().cpu().numpy())
        else:
            return [], [], []


    def _get_best_rotation(self, dict_keypoints1, dict_keypoints2, dict_descriptors1, dict_descriptors2):
        num_matches = np.zeros(4)
        keypoints1   = torch.from_numpy(  dict_keypoints1['0']).to(self.device)
        descriptors1 = torch.from_numpy(dict_descriptors1['0']).to(self.device)
        for r in range(4):
            key = str(r)
            _, distances = self._match_image_pair(  keypoints1, 
                                                    torch.from_numpy(dict_keypoints2[key]).to(self.device), 
                                                    descriptors1, 
                                                    torch.from_numpy(dict_descriptors2[key]).to(self.device))
            num_matches[r] = len(distances)

        max_matches = num_matches.max()
        mask = [True]*4
        mask[num_matches.argmax()] = False
        avg_matches = num_matches[mask].mean()
        # print(num_matches, max_matches, avg_matches)
        if max_matches > 2*avg_matches:
            # print('rotation detected', num_matches.argmax())
            return num_matches.argmax()
        else:
            return 0



    def _get_coordinates(self, keypoints, pairs, idx_pair):
        return np.array([keypoints[pair[idx_pair]] for pair in pairs])


    def _match_image_pair(self, keypoints1, keypoints2, descriptors1, descriptors2):
        with torch.inference_mode():
            distances, indice_matches = self.matcher(
                descriptors1, 
                descriptors2, 
                KF.laf_from_center_scale_ori(keypoints1[None]),
                KF.laf_from_center_scale_ori(keypoints2[None]),
            )
        return indice_matches, distances

    

    def _load_torch_image(self, path):
        img = K.io.load_image(path, K.io.ImageLoadType.RGB32, device=torch.device('cpu')).unsqueeze(0)
        return img