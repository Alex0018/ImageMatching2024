import numpy as np
import shutil
import h5py

from src.IMC_utils import IMC_Utils
from src.scene_reconstructor import SceneReconstructor


import random

random_seed = 42  
random.seed(random_seed)
np.random.seed(random_seed)


class IMC_Pipeline:

    def __init__(self, 
                 image_pairs_matcher,
                 keypoint_matcher,
                 keypoint_parser,
                 filename_matched_coordinates = 'keypoints_matched_coordinates.h5',
                 embedding_filename = 'embeddings_dinov2_base.h5', 
                 ):    

        self.embedding_filename = embedding_filename
        self.filename_matched_coordinates = filename_matched_coordinates
        self.index_pairs_filename = 'index_pairs.npy'

        self.image_pairs_matcher = image_pairs_matcher
        self.keypoint_matcher = keypoint_matcher
        self.keypoint_parser = keypoint_parser



    def run(self, dir_features, 
                  dir_images, 
                  database_path,
                  verbose=True,
                  save_intermediate_files=True):
        
        self._set_paths(dir_features, dir_images, database_path)

        image_paths, embeddings = self._process_embeddings()
        if image_paths is None:
            return [], [], [], []
        
        index_pairs = self._get_index_pairs(embeddings)
        self._process_keypoints(image_paths, index_pairs)

        map = self._get_reconstruction_map(verbose=verbose)
        print(map)
        registered_image_paths, rotation_matrice, translation_vectors = self._get_cameras_info(map)

        self._finalize_intermediate_files(save_intermediate_files)

        return image_paths, registered_image_paths, rotation_matrice, translation_vectors


    def _get_cameras_info(self, map):
        if map is None:
            return [], [], []
        
        n_images = len(map.images.items())
        registered_image_paths = np.empty(n_images, dtype='object')
        rotation_matrice = np.empty(n_images, dtype='object')
        translation_vectors = np.empty(n_images, dtype='object')

        for i, (_, im) in enumerate(map.images.items()):
            registered_image_paths[i] = str(self.dir_images / im.name)
            rotation_matrice[i]       = IMC_Utils.arr_to_str( im.cam_from_world.rotation.matrix().reshape(-1) )
            translation_vectors[i]    = IMC_Utils.arr_to_str( np.array(im.cam_from_world.translation).reshape(-1) )

        return registered_image_paths, rotation_matrice, translation_vectors



    def _get_reconstruction_map(self, verbose=False):
        maps = SceneReconstructor.run(self.dir_images, self.dir_features, self.database_path)
        
        images_registered  = 0
        best_idx = None
        for idx, rec in maps.items():
            if verbose: print(idx, rec.summary())

            if len(rec.images) > images_registered:
                images_registered = len(rec.images)
                best_idx = idx

        return maps[best_idx] if best_idx is not None else None



    def _process_keypoints(self, image_paths, index_pairs):
        if not (self.dir_features / self.filename_matched_coordinates).exists():
            self.keypoint_matcher.run(image_paths, 
                                      index_pairs, 
                                      self.dir_features, 
                                      self.filename_matched_coordinates)

        self.keypoint_parser.run(self.dir_images, 
                                 self.dir_features, 
                                 self.dir_features / self.filename_matched_coordinates,)


    def _get_index_pairs(self, embeddings):
        index_pairs = self.image_pairs_matcher.run(embeddings)
        np.save(self.dir_features / self.index_pairs_filename, index_pairs, allow_pickle=False)
        return index_pairs


    def _process_embeddings(self):
        embedding_path = self.dir_features / self.embedding_filename
        if not embedding_path.exists():
            return None, None

        with h5py.File(embedding_path, mode='r') as f_embeddings:
            image_names = [img_name for img_name in f_embeddings.keys()]
            embeddings  = [f_embeddings[key][:] for key in image_names]
            image_paths = [self.dir_images / img_name for img_name in image_names]

        return image_paths, embeddings
    

    def title(self):
        return f'{self.image_pairs_matcher.title()}_{self.keypoint_matcher.title()}_{self.keypoint_parser.title()}'


    def _finalize_intermediate_files(self, save_files):

        paths = [self.index_pairs_filename, self.filename_matched_coordinates,]

        if save_files:
            dir_experiment = self.dir_features / self.title()
            dir_experiment.mkdir(parents=True, exist_ok=True)

            for file in paths:
                shutil.move(self.dir_features / file, dir_experiment / file)
        else:
             # delete all files in paths
             pass
        

    def _set_paths(self, dir_features, dir_images, database_path):
        self.dir_features = dir_features
        self.dir_images = dir_images
        self.database_path = database_path



