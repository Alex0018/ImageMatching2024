import pandas as pd
import numpy as np

import os
from pathlib import Path
from typing import List


import random

random_seed = 42  
random.seed(random_seed)
np.random.seed(random_seed)


class IMC_Utils:
    
    def __init__(self):
        pass

    @staticmethod
    def get_metadata(df):
        '''
        returns list of (scene, dataset, image_paths)
        '''
        datasets = df['dataset'].unique()
        data = []
        for dataset in datasets:
            scenes = df.loc[df['dataset'] == dataset, 'scene'].unique()
            for scene in scenes:
                df_scene = df.loc[(df['dataset'] == dataset) & (df['scene'] == scene)]
                data.append((scene, dataset, df_scene['image_path'].values))
        return data    


    @staticmethod
    def get_storage_paths(dir_root, dataset, image_paths):
        dir_features = dir_root / f'{dataset}' / 'features'
        dir_features.mkdir(parents=True, exist_ok=True)

        dir_images = IMC_Utils.get_image_dir_paths(image_paths)

        database_path = dir_features / 'colmap.db'
        if database_path.exists():
            database_path.unlink()

        return dir_features, dir_images, database_path

    

    @staticmethod
    def get_image_dir_paths(image_paths: List[str]):
        dirs = list(set([os.path.join(*path.split('\\')[:-1]) for path in image_paths]))
        if len(dirs) != 1:
            raise 'More than one directory with images'
        return Path(dirs[0])


    
    
    @staticmethod
    def arr_to_str(a):
        return ";".join([str(x) for x in a.reshape(-1)])
    

    @staticmethod
    def get_validation_data(dataset, path_target):
        df_val = pd.read_csv(path_target)
        df_val = df_val.loc[df_val['dataset'] == dataset]
        df_val['image_path'] = f'data\\train\\{dataset}\\images\\' + df_val['image_name']
        return df_val[['image_path', 'dataset', 'scene', 'rotation_matrix', 'translation_vector']]


    @staticmethod
    def create_submission_file(dataset, scene, registered_image_paths, all_image_paths, rs, ts):
        df_sub = pd.DataFrame(index=[str(k) for k in all_image_paths])
        df_sub['rotation_matrix'] = '0;'*8 + '0'
        df_sub['translation_vector'] = '0;0;nan'
        df_sub['dataset'] = dataset
        df_sub['scene'] = scene

        if len(registered_image_paths) > 0:
            df_res = pd.DataFrame(index=[str(k) for k in registered_image_paths])
            df_res['rotation_matrix'] = rs
            df_res['translation_vector'] = ts
            df_sub.loc[df_res.index, ['rotation_matrix', 'translation_vector']] = df_res[['rotation_matrix', 'translation_vector']]

        return df_sub.reset_index(names='image_path')