import os
from prohmr.utils.geometry import batch_rodrigues
import cv2
import torch
import numpy as np
import pickle
from tqdm import tqdm

from prohmr.models import SMPL
from prohmr.configs import prohmr_config, dataset_config

def preprocess_3dpw(dataset_path: str, out_file: str):
    '''
    Generate 3DPW dataset files
    Args:
        dataset_path (str): Path to 3DPW root folder
        out_file (str): Output filename
    '''
    cfg = prohmr_config()
    smpl_cfg = {k.lower(): v for k,v in dict(cfg.SMPL).items()}
    smpl_cfg.pop('gender')

    smpl_male = SMPL(**smpl_cfg, gender='male')
    smpl_female = SMPL(**smpl_cfg, gender='female')


    # scale factor
    scaleFactor = 1.2
    coco_18_to_25 = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

    # structs we use
    imgnames_, scales_, centers_, parts_ = [], [], [], []
    poses_, betas_, genders_ = [], [], []
    body_keypoints_2d_ = []
    extra_keypoints_3d_ = []

    # get a list of .pkl files in the directory
    dataset_path = os.path.join(dataset_path, 'sequenceFiles', 'test')
    files = [os.path.join(dataset_path, f)
        for f in os.listdir(dataset_path) if f.endswith('.pkl')]
    # go through all the .pkl files
    for filename in tqdm(files):
        with open(filename, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            smpl_pose = data['poses']
            smpl_betas = data['betas']
            poses2d = data['poses2d']
            global_poses = data['cam_poses']
            genders = data['genders']
            valid = np.array(data['campose_valid']).astype(np.bool)
            num_people = len(smpl_pose)
            num_frames = len(smpl_pose[0])
            seq_name = str(data['sequence'])
            img_names = np.array(['imageFiles/' + seq_name + '/image_%s.jpg' % str(i).zfill(5) for i in range(num_frames)])
            # get through all the people in the sequence
            for i in range(num_people):
                valid_pose = smpl_pose[i][valid[i]]
                valid_betas = np.tile(smpl_betas[i][:10].reshape(1,-1), (num_frames, 1))
                valid_betas = valid_betas[valid[i]]
                valid_keypoints_2d = poses2d[i][valid[i]]
                valid_img_names = img_names[valid[i]]
                valid_global_poses = global_poses[valid[i]]
                gender = genders[i]
                # consider only valid frames
                for valid_i in range(valid_pose.shape[0]):
                    part = valid_keypoints_2d[valid_i,:,:].T.copy()
                    part = part[part[:,2]>0,:]
                    bbox = [min(part[:,0]), min(part[:,1]),
                        max(part[:,0]), max(part[:,1])]
                    center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
                    scale = scaleFactor*max(bbox[2]-bbox[0], bbox[3]-bbox[1])

                    # transform global pose
                    pose = valid_pose[valid_i]
                    extrinsics = valid_global_poses[valid_i][:3,:3]
                    pose[:3] = cv2.Rodrigues(np.dot(extrinsics, cv2.Rodrigues(pose[:3])[0]))[0].T[0]
                    body_keypoints_2d = np.zeros((25, 3))
                    body_keypoints_2d[coco_18_to_25] = valid_keypoints_2d[valid_i, :, :].T
                    rotmat = batch_rodrigues(torch.tensor(pose, dtype=torch.float32).reshape(-1, 3)).unsqueeze(0)
                    if gender == 'm':
                        extra_keypoints_3d = smpl_male(global_orient=rotmat[:, [0]],
                                                       body_pose=rotmat[:, 1:],
                                                       betas=torch.tensor(valid_betas[valid_i], dtype=torch.float32).unsqueeze(0)).joints[0,25:]
                    else:
                        extra_keypoints_3d = smpl_female(global_orient=rotmat[:, [0]],
                                                         body_pose=rotmat[:, 1:],
                                                         betas=torch.tensor(valid_betas[valid_i], dtype=torch.float32).unsqueeze(0)).joints[0,25:]
                    extra_keypoints_3d = torch.cat((extra_keypoints_3d, torch.ones(extra_keypoints_3d.shape[0], 1)), dim=-1).numpy()

                    imgnames_.append(valid_img_names[valid_i])
                    centers_.append(center)
                    scales_.append(scale)
                    poses_.append(pose)
                    body_keypoints_2d_.append(body_keypoints_2d)
                    betas_.append(valid_betas[valid_i])
                    extra_keypoints_3d_.append(extra_keypoints_3d)
                    genders_.append(gender)
    np.savez(out_file,
             imgname=imgnames_,
             center=centers_,
             scale=scales_,
             body_pose=poses_,
             has_body_pose=np.ones(len(poses_)),
             betas=betas_,
             has_betas=np.ones(len(betas_)),
             body_keypoints_2d=body_keypoints_2d_,
             extra_keypoints_3d=extra_keypoints_3d_,
             gender=genders_)

if __name__ == '__main__':
    dataset_cfg = dataset_config()['3DPW-TEST']
    preprocess_3dpw(dataset_cfg.IMG_DIR, dataset_cfg.DATASET_FILE)
