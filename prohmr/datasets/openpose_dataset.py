"""
Create an ImageDataset on the fly from a collections of images with corresponding OpenPose detections.
In order to preserve backwards compatibility with SMPLify-X, parts of the code are adapted from
https://github.com/vchoutas/smplify-x/blob/master/smplifyx/data_parser.py
"""
import os
import json
import numpy as np
from typing import Dict, Optional

from yacs.config import CfgNode

from .image_dataset import ImageDataset

def read_openpose(keypoint_fn: str, max_people_per_image: Optional[int] = None):
    with open(keypoint_fn) as keypoint_file:
        data = json.load(keypoint_file)

    body_keypoints_2d = []

    for i, person in enumerate(data['people']):
        if max_people_per_image is not None and i >= max_people_per_image:
            break
        openpose_detections = np.array(person['pose_keypoints_2d']).reshape(-1, 3)
        body_keypoints_2d.append(openpose_detections)

    return body_keypoints_2d

class OpenPoseDataset(ImageDataset):

    def __init__(self,
                 cfg: CfgNode,
                 img_folder: str,
                 keypoint_folder: str,
                 rescale_factor: float = 1.2,
                 train: bool = False,
                 max_people_per_image: Optional[int] = None,
                 **kwargs):
        """
        Dataset class used for loading images and corresponding annotations from image/OpenPose pairs.
        It builds and ImageDataset on-the-fly instead of reading the data from an npz file.
        Args:
            cfg (CfgNode): Model config file.
            img_folder (str): Folder containing images.
            keypoint_folder (str): Folder containing OpenPose detections.
            rescale_factor (float): Scale factor for rescaling bounding boxes computed from the OpenPose keypoints.
            train (bool): Whether it is for training or not (enables data augmentation)
        """

        self.cfg = cfg
        self.img_folder = img_folder
        self.keypoint_folder = keypoint_folder

        self.img_paths = [os.path.join(self.img_folder, img_fn)
                          for img_fn in os.listdir(self.img_folder)]
        self.img_paths = sorted(self.img_paths)
        self.rescale_factor = rescale_factor
        self.train = train
        self.max_people_per_image = max_people_per_image
        self.img_dir = ''
        body_permutation = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]
        extra_permutation = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18]
        flip_keypoint_permutation = body_permutation + [25 + i for i in extra_permutation]
        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(self.cfg.MODEL.IMAGE_STD)
        self.flip_keypoint_permutation = flip_keypoint_permutation
        self.preprocess()

    def preprocess(self):
        """
        Preprocess annotations and convert them to the format ImageDataset expects.
        """
        body_keypoints = []
        imgnames = []
        scales = []
        centers = []
        for i in range(len(self.img_paths)):
            img_path = self.img_paths[i]
            item = self.get_example(img_path)
            num_people = item['keypoints_2d'].shape[0]
            for n in range(num_people):
                keypoints_n = item['keypoints_2d'][n]
                keypoints_valid_n = keypoints_n[keypoints_n[:, 1] > 0, :].copy()
                bbox = [min(keypoints_valid_n[:,0]), min(keypoints_valid_n[:,1]),
                    max(keypoints_valid_n[:,0]), max(keypoints_valid_n[:,1])]
                center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
                scale = self.rescale_factor*max(bbox[2]-bbox[0], bbox[3]-bbox[1])
                body_keypoints.append(keypoints_n)
                scales.append(scale)
                centers.append(center)
                imgnames.append(item['img_path'])
        self.imgname = np.array(imgnames)
        self.scale = np.array(scales).astype(np.float32) / 200.0
        self.center = np.array(centers).astype(np.float32)
        body_keypoints_2d = np.array(body_keypoints).astype(np.float32)
        extra_keypoints_2d = np.zeros((len(self.center), 19, 3))
        self.keypoints_2d = np.concatenate((body_keypoints_2d, extra_keypoints_2d), axis=1).astype(np.float32)
        body_keypoints_3d = np.zeros((len(self.center), 25, 4), dtype=np.float32)
        extra_keypoints_3d = np.zeros((len(self.center), 19, 4), dtype=np.float32)
        self.keypoints_3d = np.concatenate((body_keypoints_3d, extra_keypoints_3d), axis=1).astype(np.float32)
        num_pose = 3 * (self.cfg.SMPL.NUM_BODY_JOINTS + 1)
        self.body_pose = np.zeros((len(self.imgname), num_pose), dtype=np.float32)
        self.has_body_pose = np.zeros(len(self.imgname), dtype=np.float32)
        self.betas = np.zeros((len(self.imgname), 10), dtype=np.float32)
        self.has_betas = np.zeros(len(self.imgname), dtype=np.float32)

    def get_example(self, img_path: str) -> Dict:
        """
        Load an image and corresponding OpenPose detections.
        Args:
            img_path (str): Path to image file.
        Returns:
            Dict: Dictionary containing the image path and 2D keypoints if available, else an empty dictionary.
        """
        img_fn, _ = os.path.splitext(os.path.split(img_path)[1])

        keypoint_fn = os.path.join(self.keypoint_folder,
                               img_fn + '_keypoints.json')
        keypoints_2d = read_openpose(keypoint_fn, max_people_per_image=self.max_people_per_image)

        if len(keypoints_2d) < 1:
            return {}
        keypoints_2d = np.stack(keypoints_2d)

        item = {'img_path': img_path,
                'keypoints_2d': keypoints_2d}
        return item
