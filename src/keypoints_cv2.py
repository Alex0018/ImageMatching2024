# Reference: 
# https://www.kaggle.com/code/motono0223/imc-2024-multi-models-pipeline/notebook


import numpy as np

import torch

from tqdm import tqdm
import h5py
import cv2



from src.keypoints_matcher_base import KeypointsMatcherBase




class Keypoints_cv2(KeypointsMatcherBase):

    FILENAME_KEYPOINTS = 'keypoints_cv2.h5'
    FILENAME_KEYPOINTS_DESCRIPTORS = 'descriptors_cv2.h5'
    FILENAME_KEYPOINTS_FAST = 'keypoints_fast.h5'
    FILENAME_KEYPOINTS_DESCRIPTORS_FAST = 'descriptors_fast.h5'

    dict_extractors = {'orb':   cv2.ORB_create,
                       'akaze': cv2.AKAZE_create}

    def __init__(self, 
                 extractor_class_label='orb',
                 resize_small_edge_to=1024,
                 device=torch.device('cpu')):
        super(Keypoints_cv2, self).__init__()
        
        self.device = device

        self.extractor_class_label = extractor_class_label
        self.resize_small_edge_to = resize_small_edge_to

        self.extractor_dtype = torch.float32 # ALIKED has issues with float16
        extractor_class = Keypoints_cv2.dict_extractors.get(extractor_class_label)
        self.extractor = extractor_class()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


    def title(self):
        return f'{self.extractor_class_label}{self.resize_small_edge_to}_cv2BF'
    

    def get_params_dict(self):
        return {'extractor_class_label': self.extractor_class_label,
                'resize_small_edge_to': self.resize_small_edge_to}


    def detect_keypoints(self, image_paths, dir_features):
        keypoints_dict = {}
        descriptors_dict = {}

        for path in tqdm(image_paths, desc='Computing keypoints'):             
            with torch.inference_mode():               
                image, image_shape = self._load_torch_image(path)

                key = path.name
                keypoints_dict[key] = {}
                descriptors_dict[key] = {}
                for rotation in range(4):
                    img = np.rot90(image, rotation, axes=[0,1])
                    # print(img.shape)
                    keypoints, descriptors = self.extractor.detectAndCompute(img, None)
                    keypoints_dict[key][rotation]   = keypoints
                    descriptors_dict[key][rotation] = descriptors

        
        return keypoints_dict, descriptors_dict




    def run(self, image_paths, index_pairs, dir_features, filename_save, min_matches=15):

        count_matched_pairs = 0

        keypoints_dict, descriptors_dict = self.detect_keypoints(image_paths, dir_features)

        image_shapes = [cv2.imread(str(path)).shape[:2] for path in image_paths]
        matches = {path.name:{} for path in image_paths}
           
        for (idx1, idx2, _) in tqdm(index_pairs, desc='Matching keypoints'):

            key1, key2 = image_paths[idx1].name, image_paths[idx2].name
            image1_original_shape = image_shapes[idx1] 
            image2_original_shape = image_shapes[idx2] 

            keypoints1, keypoints2, confidences = self._match_image_pair_with_rotation(
                                                            keypoints_dict, descriptors_dict, 
                                                            key1, key2, 
                                                            image1_original_shape, image2_original_shape)            

            if len(confidences) >= min_matches:
                keypoints1, keypoints2 = self._postprocess_keypoints(keypoints1, keypoints2, image1_original_shape, image2_original_shape)

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


    def _match_image_pair_with_rotation(self, 
                                        keypoints_dict,        descriptors_dict, 
                                        key1,                  key2, 
                                        image1_original_shape, image2_original_shape):
        
        rot2 = self._get_best_rotation(descriptors_dict[key1], 
                                       descriptors_dict[key2], )   
        keypoints1 = keypoints_dict[key1][0]
        keypoints2 = keypoints_dict[key2][rot2]
        matches = self._match_image_pair(descriptors_dict[key1][0], descriptors_dict[key2][rot2])
        
        distances = [m.distance for m in matches]
        keypoints_coordinates_1 = [keypoints1[m.queryIdx].pt for m in matches]
        keypoints_coordinates_2 = [keypoints2[m.trainIdx].pt for m in matches]

        if  (len(keypoints_coordinates_1) > 0) and \
            (len(keypoints_coordinates_2) > 0):
            return (np.array(keypoints_coordinates_1), 
                    np.array(KeypointsMatcherBase._rotate_keypoints(keypoints_coordinates_2, *image2_original_shape, rot2)),
                    np.array(distances).reshape(-1, 1))
        else:
            return [], [], []


    def _get_best_rotation(self, dict_descriptors1, dict_descriptors2):
        num_matches = np.zeros(4)
        descriptors1 = dict_descriptors1[0]
        for r in range(4):
            matches = self._match_image_pair(descriptors1, dict_descriptors2[r])
            num_matches[r] = len(matches)

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


    def _match_image_pair(self, descriptors1, descriptors2):
        matches = self.matcher.match(descriptors1, descriptors2)
        return matches


    

    def _postprocess_keypoints(self, keypoints1, keypoints2, orig_shape_1, orig_shape_2):
        ratio1 = self.resize_small_edge_to / min(orig_shape_1[0], orig_shape_1[1])
        keypoints1 /= ratio1

        ratio2 = self.resize_small_edge_to / min(orig_shape_2[0], orig_shape_2[1])
        keypoints2 /= ratio2
        return keypoints1, keypoints2



    def _read_image_with_resize(self, path):
        img = cv2.imread(str(path))
        original_shape = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ratio = self.resize_small_edge_to / min(img.shape[0], img.shape[1])
        img = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
        return img, original_shape


    def _load_torch_image(self, path):
        img_resized, original_shape = self._read_image_with_resize(path)
        return img_resized, original_shape
