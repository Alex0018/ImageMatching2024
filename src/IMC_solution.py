import torch
import wandb

from datetime import datetime

from src.imc24 import score

from src.styles import TXT_ACC, TXT_RESET

from src.parser_matched_keypoints import ParserMatchedKeypoints
from src.image_pairs_matcher import ImagePairsMatcher

from src.keypoints_LoFTR import Keypoints_LoFTR
from src.keypoints_LightGlue import Keypoints_LightGlue
from src.keypoints_cv2 import Keypoints_cv2

from src.IMC_utils import IMC_Utils
from src.IMC_pipeline import IMC_Pipeline


PROJECT = 'ImageMatching2024'


class IMC_Solution:

    keypoint_matcher_class_dict = {'loftr': Keypoints_LoFTR,
                                   'LightGlue':  Keypoints_LightGlue,
                                   'cv2_brute_force': Keypoints_cv2,}

    def __init__(self, 
                 embedding_filename = 'embeddings_dinov2_base.h5', 
                 image_pairs_matcher_args={},
                 keypoint_matcher_class_label: str = 'loftr',
                 keypoint_matcher_args={},
                 keypoint_parser_args={},
                 device=torch.device('cpu')):
        self.init_successful = True        

        self.embedding_filename = embedding_filename
        self.index_pairs_filename = 'index_pairs.npy'

        self.image_pairs_matcher = ImagePairsMatcher(**image_pairs_matcher_args)
        
        keypoint_matcher_class = IMC_Solution.keypoint_matcher_class_dict.get(keypoint_matcher_class_label)
        if keypoint_matcher_class is None:
            self.init_successful = False
            return

        self.keypoint_matcher = keypoint_matcher_class(device=device, **keypoint_matcher_args)
        self.keypoint_parser = ParserMatchedKeypoints(**keypoint_parser_args)

        self.processor = IMC_Pipeline(  image_pairs_matcher=self.image_pairs_matcher,
                                        keypoint_matcher=self.keypoint_matcher,
                                        keypoint_parser=self.keypoint_parser,
                                        filename_matched_coordinates=f'keypoints_{self.keypoint_matcher.title()}.h5')
        
        self.config = { 'matcher': keypoint_matcher_class_label,
                        'pairs_params': self.image_pairs_matcher.get_params_dict(),
                        'matcher_params': self.keypoint_matcher.get_params_dict(),
                        'parser_params': self.keypoint_parser.get_params_dict(),
                        }



    def run(self, metadata, dir_root, path_target, verbose=True):

        wandb_run = wandb.init(project=PROJECT, config=self.config, name=self.processor.title())

        for (scene, dataset, paths) in metadata:

            if verbose: print(f'{TXT_ACC} {dataset} {TXT_RESET}')

            start_time = datetime.now()

            dir_features, dir_images, database_path = IMC_Utils.get_storage_paths(dir_root, dataset, paths)
            
            image_paths, \
            registered_image_paths, \
            rotation_matrice, \
            translation_vectors = self.processor.run(   dir_features=dir_features, 
                                                        dir_images=dir_images, 
                                                        database_path=database_path,
                                                        verbose=verbose,)
            
            df_val = IMC_Utils.get_validation_data(dataset, path_target)
            df_res = IMC_Utils.create_submission_file(dataset, 
                                                      scene, 
                                                      registered_image_paths, 
                                                      image_paths, 
                                                      rotation_matrice, 
                                                      translation_vectors)
            sc = score(df_val, df_res)
            wandb_run.log({f'{dataset}_score': sc, 
                           f'{dataset}_nreg_images': len(registered_image_paths),
                           f'{dataset}_time': str(datetime.now() - start_time),
                        }) 
        
            df_res.to_csv(dir_root / f'result_{dataset}_{self.processor.title()}.csv', index=False)
        
        wandb.finish()
            
