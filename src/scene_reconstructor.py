from pathlib import Path
import pycolmap

from src.database import COLMAPDatabase 
from src.h5_to_db import add_keypoints, add_matches


class SceneReconstructor:

    # def __init__(self):
    #     pass


    @staticmethod
    def run(dir_images, dir_features, database_path):
        
        SceneReconstructor._import_keypoints_into_colmap(
                        dir_images, 
                        dir_features, 
                        database_path,
                    )

        pycolmap.match_exhaustive(database_path, sift_options={'num_threads':1})

        mapper_options = pycolmap.IncrementalPipelineOptions(**{
            'min_model_size': 3, # By default colmap does not generate a reconstruction if less than 10 images are registered. 
            'max_num_models': 10,
            'num_threads':1,
        })

        output_path = dir_features / "colmap_rec"
        output_path.mkdir(parents=True, exist_ok=True)

        # Incrementally start reconstructing the scene (sparse reconstruction)
        # The process starts from a random pair of images and is incrementally extended by 
        # registering new images and triangulating new points.
        maps = pycolmap.incremental_mapping(
            database_path=database_path, 
            image_path=dir_images,
            output_path=output_path, 
            options=mapper_options,
        )

        return maps

    @staticmethod
    def _import_keypoints_into_colmap(
                path: Path,
                feature_dir: Path,
                database_path: str = "colmap.db",
            ) -> None:

        db = COLMAPDatabase.connect(database_path)
        db.create_tables()
        single_camera = False
        fname_to_id = add_keypoints(db, feature_dir, path, "", "simple-pinhole", single_camera)
        add_matches(
            db,
            feature_dir,
            fname_to_id,
        )
        db.commit()


