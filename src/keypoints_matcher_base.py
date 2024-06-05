
# Reference: 
# https://www.kaggle.com/code/motono0223/imc-2024-multi-models-pipeline/notebook

from abc import ABC, abstractmethod
import numpy as np

import torch
from torchvision.transforms import Resize
import cv2
from tqdm import tqdm
import h5py

import gc

'''
abstract base class for keypoints matcher
'''

class KeypointsMatcherBase(ABC):

    MIN_MATCHES = 500

    def __init__(self):
        pass


    def run(self, image_paths, index_pairs, dir_features, filename_save, min_matches=15):
        '''
        constraint: up to 100 images per scene so that they fit into memory
        '''
        count_matched_pairs = 0

        images_with_shapes = [self._load_torch_image(path) for path in image_paths]
        images_small = [Resize(128, antialias=True)(img[0]) for img in images_with_shapes]

        with h5py.File(dir_features / filename_save, mode='w') as f_match:    
            for (idx1, idx2, _) in tqdm(index_pairs, desc='Matching keypoints'):

                image1, orig_shape_1 = images_with_shapes[idx1]
                image2, orig_shape_2 = images_with_shapes[idx2]
                small_image1 = images_small[idx1]
                small_image2 = images_small[idx2]

                keypoints1, keypoints2, confidences = self._match_image_pair_with_rotation(image1, image2, small_image1, small_image2)
                
                keypoints1, keypoints2 = self._postprocess_keypoints(keypoints1, keypoints2, image1, image2, orig_shape_1, orig_shape_2)

                if confidences.shape[0] >= min_matches:
                    group  = f_match.require_group(image_paths[idx1].name)
                    group.create_dataset(image_paths[idx2].name, 
                                         data=np.concatenate([keypoints1, keypoints2, confidences], axis=1).astype(np.float32))
                    count_matched_pairs += 1
                
                del image1, image2, keypoints1, keypoints2, confidences
                torch.cuda.empty_cache()
                gc.collect()

        return count_matched_pairs
    


    def _postprocess_keypoints(self, keypoints1, keypoints2, image1, image2, orig_shape_1, orig_shape_2):
        return keypoints1, keypoints2



    def _get_best_rotation(self, image1, image2):
        num_matches = np.zeros(4)
        for r in range(4):
            rotated_image2 = image2.rot90(r, dims=[2,3])
            _, _, confidences = self._match_image_pair(image1, rotated_image2)
            num_matches[r] = len(confidences)

        max_matches = num_matches.max()
        mask = [True]*4
        mask[num_matches.argmax()] = False
        avg_matches = num_matches[mask].mean()
        # print(num_matches, max_matches, avg_matches)
        if max_matches > 2*avg_matches:
            # if num_matches.argmax() > 0:
                # print('rotation detected', num_matches.argmax(), '\n', num_matches, max_matches, avg_matches)
            return num_matches.argmax()
        else:
            return 0


    
    def _match_image_pair_with_rotation(self, image1, image2, small_image1, small_image2):
        rot2 = self._get_best_rotation(small_image1, small_image2)

        rotated_image2 = image2.rot90(rot2, dims=[2,3])
        keypoints_coordinates_1, keypoints_coordinates_2, confidences = self._match_image_pair(image1, rotated_image2)

        return (keypoints_coordinates_1, 
                KeypointsMatcherBase._rotate_keypoints(keypoints_coordinates_2, *image2.shape[2:], rot2),
                confidences.reshape(-1, 1))


    @abstractmethod
    def _match_image_pair(self, image1, image2):
        '''
        returns numpy arrays (keypoints_coordinates_1, keypoints_coordinates_2, confidences)

        - keypoints_coordinates_1.shape = (number of matches, 2)
        - keypoints_coordinates_2.shape = (number of matches, 2)
        - confidences.shape = (number of matches, 1)
        '''
        pass


    @abstractmethod
    def _load_torch_image(self, path):
        '''
        returns (image, image_original_shape)

        - returned image can be resized
        '''
        pass


    @abstractmethod
    def title(self):
        pass

    @abstractmethod
    def get_params_dict(self):
        pass


    @staticmethod
    def _rotate_keypoints(keypoints, h, w, angle):
        if angle == 0:
            return keypoints
        elif angle == 1:
            rot_keypoints = np.zeros(keypoints.shape)
            rot_keypoints[:, 0] = w - keypoints[:, 1]
            rot_keypoints[:, 1] = keypoints[:, 0]
            return rot_keypoints
        elif angle == 2:
            rot_keypoints = np.zeros(keypoints.shape)
            rot_keypoints[:, 0] = w - keypoints[:, 0]
            rot_keypoints[:, 1] = h - keypoints[:, 1]
            return rot_keypoints
        elif angle == 3:
            rot_keypoints = np.zeros(keypoints.shape)
            rot_keypoints[:, 0] = keypoints[:, 1]
            rot_keypoints[:, 1] = h - keypoints[:, 0]
            return rot_keypoints
    

