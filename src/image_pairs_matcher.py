import numpy as np
from scipy.spatial.distance import cdist 


class ImagePairsMatcher:

    def __init__(self,      threshold:   float = 0.6,
                            min_matches: int = 20,):
        self.threshold = threshold
        self.min_matches = min_matches              
        
        
    def run(self, embeddings: np.array):       
        '''
        returns list of (index_image1, index_image2, distance) sorted by distance
        '''
        matches = []
        
        distances = cdist(embeddings, embeddings)
        
        mask = distances <= self.threshold
        image_indices = np.arange(len(embeddings))
        
        for idx1 in range(len(embeddings)):
            mask_row = mask[idx1]
            indices_to_match = image_indices[mask_row]
            
            if len(indices_to_match) < self.min_matches:
                indices_to_match = np.argsort(distances[idx1])[:self.min_matches]
                
            for idx2 in indices_to_match:
                if idx2 <= idx1:
                    continue                    
                matches.append((idx1, idx2, distances[idx1, idx2]))
                    
        return sorted(matches, key=lambda x: x[2])
    

    def title(self):
        return f'pairs{self.threshold}x{self.min_matches}'
    

    def get_params_dict(self):
        return {'thresholds': self.threshold,
                'min_matches': self.min_matches}
