# Reference: 
# https://www.kaggle.com/code/motono0223/imc-2024-multi-models-pipeline/notebook


import kornia as K
from kornia.feature import LoFTR
import torch
import cv2
from src.keypoints_matcher_base import KeypointsMatcherBase



class Keypoints_LoFTR(KeypointsMatcherBase):

    def __init__(self, device=torch.device('cpu'), resize_small_edge_to=512):
        super(Keypoints_LoFTR, self).__init__()
        self.device = device

        self.matcher = LoFTR(pretrained=None).to(self.device)
        self.matcher.load_state_dict(torch.load('src/loftr_outdoor.ckpt')['state_dict'])
        self.matcher.eval()     

        self.resize_small_edge_to = resize_small_edge_to


    def title(self):
        return f'LoFTR{self.resize_small_edge_to}'
    

    def get_params_dict(self):
        return {'resize_small_edge_to': self.resize_small_edge_to}


    def _match_image_pair(self, image1, image2):
        with torch.no_grad():
            correspondences = self.matcher( {"image0": image1.to(self.device),"image1": image2.to(self.device)} )
            keypoints1 = correspondences['keypoints0'].cpu().numpy()
            keypoints2 = correspondences['keypoints1'].cpu().numpy()
            confidences = correspondences['confidence'].cpu().numpy()
        return keypoints1, keypoints2, confidences
    

    def _postprocess_keypoints(self, keypoints1, keypoints2, image1, image2, orig_shape_1, orig_shape_2):
        keypoints1[:,0] *= (float(orig_shape_1[1]) / float(image1.shape[3]))
        keypoints1[:,1] *= (float(orig_shape_1[0]) / float(image1.shape[2]))

        keypoints2[:,0] *= (float(orig_shape_2[1]) / float(image2.shape[3]))
        keypoints2[:,1] *= (float(orig_shape_2[0]) / float(image2.shape[2]))
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
        img_resized = K.image_to_tensor(img_resized, False).float() /255.
        img_resized = K.color.bgr_to_rgb(img_resized)
        img_resized = K.color.rgb_to_grayscale(img_resized)
        return img_resized, original_shape
