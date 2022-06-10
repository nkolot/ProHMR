import os
import sys
import cv2
import glob
import h5py
import numpy as np
import argparse
from spacepy import pycdf
import pickle

from prohmr.configs import prohmr_config, dataset_config

parser = argparse.ArgumentParser(description='Generate H36M dataset files')
parser.add_argument('--split', type=str, required=True, choices=['VAL', 'VAL-P2', 'TRAIN', 'MULTIVIEW'], help='Dataset split to preprocess')

args = parser.parse_args()

def preprocess_h36m(dataset_path: str, out_file: str, split: str, extract_img: bool = False):
    '''
    Generate H36M training and validation npz files
    Args:
        dataset_path (str): Path to H36M root
        out_file (str): Output filename
        split (str): Whether it is TRAIN/VAL/VAL-P2
        extract_img: Whether to extract the images from the videos
    '''

    # convert joints to global order
    h36m_idx = [11, 6, 7, 8, 1, 2, 3, 12, 24, 14, 15, 17, 18, 19, 25, 26, 27]
    global_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]

    # structs we use
    imgnames_, scales_, centers_, extra_keypoints_2d_, extra_keypoints_3d_  = [], [], [], [], []

    if split == 'train':
        user_list = [1, 5, 6, 7, 8]
    elif split == 'val' or split == 'val-p2':
        user_list = [9, 11]

    # go over each user
    for user_i in user_list:
        user_name = 'S%d' % user_i
        # path with GT bounding boxes
        bbox_path = os.path.join(dataset_path, user_name, 'MySegmentsMat', 'ground_truth_bb')
        # path with GT 3D pose
        pose_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures', 'D3_Positions_mono')
        # path with GT 2D pose
        pose2d_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures', 'D2_Positions')
        # path with videos
        vid_path = os.path.join(dataset_path, user_name, 'Videos')

        # go over all the sequences of each user
        seq_list = glob.glob(os.path.join(pose_path, '*.cdf'))
        seq_list.sort()
        for seq_i in seq_list:

            # sequence info
            seq_name = seq_i.split('/')[-1]
            action, camera, _ = seq_name.split('.')
            action = action.replace(' ', '_')
            # irrelevant sequences
            if action == '_ALL':
                continue

            # 3D pose file
            poses_3d = pycdf.CDF(seq_i)['Pose'][0]

            # 2D pose file
            pose2d_file = os.path.join(pose2d_path, seq_name)
            poses_2d = pycdf.CDF(pose2d_file)['Pose'][0]

            # bbox file
            bbox_file = os.path.join(bbox_path, seq_name.replace('cdf', 'mat'))
            bbox_h5py = h5py.File(bbox_file)

            # video file
            if extract_img:
                vid_file = os.path.join(vid_path, seq_name.replace('cdf', 'mp4'))
                imgs_path = os.path.join(dataset_path, 'images')
                vidcap = cv2.VideoCapture(vid_file)
                success, image = vidcap.read()

            # go over each frame of the sequence
            for frame_i in range(poses_3d.shape[0]):
                # read video frame
                if extract_img:
                    success, image = vidcap.read()
                    if not success:
                        break

                # check if you can keep this frame
                if frame_i % 5 == 0 and (split == 'VAL' or split == 'TRAIN' or camera == '60457274'):
                    # image name
                    imgname = '%s_%s.%s_%06d.jpg' % (user_name, action, camera, frame_i+1)
                    
                    # save image
                    if extract_img:
                        img_out = os.path.join(imgs_path, imgname)
                        cv2.imwrite(img_out, image)

                    # read GT bounding box
                    mask = bbox_h5py[bbox_h5py['Masks'][frame_i,0]].value.T
                    ys, xs = np.where(mask==1)
                    bbox = np.array([np.min(xs), np.min(ys), np.max(xs)+1, np.max(ys)+1])
                    center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
                    scale = 0.9*max(bbox[2]-bbox[0], bbox[3]-bbox[1])

                    # read GT 2D pose
                    partall = np.reshape(poses_2d[frame_i,:], [-1,2])
                    part17 = partall[h36m_idx]
                    extra_keypoints_2d = np.zeros([19,3])
                    extra_keypoints_2d[global_idx, :2] = part17
                    extra_keypoints_2d[global_idx, 2] = 1

                    # read GT 3D pose
                    Sall = np.reshape(poses_3d[frame_i,:], [-1,3])/1000.
                    S17 = Sall[h36m_idx]
                    S17 -= S17[0] # root-centered
                    extra_keypoints_3d = np.zeros([19,4])
                    extra_keypoints_3d[global_idx, :3] = S17
                    extra_keypoints_3d[global_idx, 3] = 1

                    # store data
                    imgnames_.append(os.path.join('images', imgname))
                    centers_.append(center)
                    scales_.append(scale)
                    extra_keypoints_2d_.append(extra_keypoints_2d)
                    extra_keypoints_3d_.append(extra_keypoints_3d)

    # store the data struct
    if not os.path.isdir(out_file):
        os.makedirs(out_file)
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       extra_keypoints_2d=extra_keypoints_2d_,
                       extra_keypoints_3d=extra_keypoints_3d_)

def preprocess_h36m_multiview(input_file: str, out_file: str):
    '''
    Generate H36M multiview evaluation file
    Args:
        input_file (str): H36M validation npz filename
        out_file (str): Output filename
    '''
    x = dict(np.load(input_file))
    imgname = x['imgname']
    actions = np.unique([img.split('/')[-1].split('.')[0] for img in imgname])
    frames = {action: {} for action in actions}
    for i, img in enumerate(imgname):
        action_with_cam = img.split('/')[-1]
        action = action_with_cam.split('.')[0]
        cam = action_with_cam.split('.')[1].split('_')[0]
        if cam in frames[action]:
            frames[action][cam].append(i)
        else:
            frames[action][cam] = []
    data_list = []
    for action in frames.keys():
        cams = list(frames[action].keys())
        for n in range(len(frames[action][cams[0]])):
            keep_frames = []
            for cam in cams:
                keep_frames.append(frames[action][cam][n])
            data_list.append({k: v[keep_frames] for k,v in x.items()})
    pickle.dump(data_list, open(out_file, 'wb'))

if __name__ == '__main__':
    dataset_cfg = dataset_config()[f'H36M-{args.split}']
    if args.split == 'MULTIVIEW':
        preprocess_h36m_multiview(dataset_config()['H36M-VAL'].DATASET_FILE, dataset_cfg.DATASET_FILE)
    else:
        preprocess_h36m(dataset_cfg.IMG_DIR, dataset_cfg.DATASET_FILE, args.split, extract_img=True)
