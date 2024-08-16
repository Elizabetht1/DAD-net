import glob
import os
import json
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

from tools.helper import compute_diver_body_frame, compute_angle_difference, noise_mask,feat_noise_mask




class BaseFeaturesDataset(Dataset):
    def __init__(self, data_path, window_size=3, sample_frequency=10,window_overlap=0.5,transform=None):
        self.window = window_size * sample_frequency
        self.sample_interval = 1 / sample_frequency

        self.class_dict = {"notmoving": 0, "moving": 1}

        self.window_overlap = window_overlap
        
        # Initialize the dataset
        self.metadata = self._load_data(data_path)

        self.seq_pairs = self._get_pairs(self.metadata)
       
        self.transform=transform

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.seq_pairs)

    def __getitem__(self, idx):
        raise NotImplementedError

    def _load_data(self, data_path):
        metadata_path = sorted(glob.glob(os.path.join(data_path, "**/**")))

        metadata = {}
        for path in metadata_path:
            target = Path(path).parent.stem
            subject = f"{target}_{Path(path).stem}"

            features_path = sorted(glob.glob(
                os.path.join(path, "features/*.npy")))
            pose_path = sorted(glob.glob(
                os.path.join(path, "pose/*.json")))

            assert len(features_path) == len(pose_path), \
                ("Number of features and pose files do not match, "
                 "features: {}, pose: {}".format(
                     len(features_path), len(pose_path)))

            metadata[subject] = {
                "data": [],
                "target": target
            }

            for i in range(len(features_path)):
                features = np.load(features_path[i])

                with open(pose_path[i], 'r') as f:
                    data = json.load(f)

                metadata[subject]["data"].append({
                    'features': features,
                    'pose_2d': np.array(data['pose2d']),
                    'pose_3d': np.array(data['pose3d'])
                })

        return metadata

    def _get_pairs(self, metadata):
        # Build lineage info (only consider stride=1 here)
        # ex: [(subject1, 0, 30), (subject1, 1, 31), (subject1, 2, 32), ...
        seq_pairs = []  # (seq_subj, start_frame, end_frame) tuples
        for subject, db in metadata.items():
            n_chunks = len(db["data"])
            bounds = np.arange(n_chunks + 1)
            stride = int((1-self.window_overlap)*self.window) ##determines degree of overlap
            seq_pairs += zip(np.repeat(subject, len(bounds - self.window)),
                             bounds[:-self.window:stride],
                             bounds[self.window::stride])

        return seq_pairs


class PoseFeaturesDataset(BaseFeaturesDataset):
    def __init__(self, data_fp, window_size=3, sample_frequency=10,window_overlap=0.5,
                 include_2d=False, include_3d=False,transform=None):
        super().__init__(data_fp, window_size, sample_frequency,window_overlap,transform)

        self.include_2d = include_2d
        self.include_3d = include_3d

        self._add_imu_features()

    def __getitem__(self, idx):

        ''' 
        output: (time_steps,features)
        '''

        seq_subject, start_3d, end_3d = self.seq_pairs[idx]
        db = self.metadata[seq_subject]

        pose_feats = []
        target = db['target']
        for d in db["data"][start_3d:end_3d]:
            feat = d['angular_acc']
            if self.include_2d:
                feat = np.concatenate([feat, d['pose_2d'].flatten()])
            if self.include_3d:
                feat = np.concatenate([feat, d['pose_3d'].flatten()])
            pose_feats.append(feat)

        pose_feats = np.array(pose_feats, dtype=np.float32)

        if self.transform:
            pose_feats = self.transform(pose_feats)

        return pose_feats.transpose(), np.array(self.class_dict[target])

    def _add_imu_features(self):
        for values in self.metadata.values():
            pose_3d = []

            for data in values["data"]:
                pose_3d.append(data['pose_3d'])

            angular_acc = self._cal_acceleration(np.array(pose_3d))
            assert len(angular_acc) == len(values["data"]), \
                ("Number of angular acceleration and pose files do not match, "
                 "angular_acc: {}, pose: {}".format(
                     len(angular_acc), len(values["data"])))

            for i, d in enumerate(values["data"]):
                d.update({"angular_acc": angular_acc[i]})

    def _cal_acceleration(self, pose_feats):
        # Calculate the angular acceleration of the body frame
        pose_feats_3d = pose_feats[..., :3]

        frame_axis = [
            compute_diver_body_frame(pose) for pose in pose_feats_3d
        ]

        frame_omega = [np.array([0, 0, 0])]  # Initial angular velocity is 0
        for i in range(1, len(frame_axis)):
            prev = frame_axis[i - 1]
            curr = frame_axis[i]

            angle_diff = compute_angle_difference(prev, curr)
            frame_omega.append(angle_diff / self.sample_interval)

        frame_omega = np.array(frame_omega, dtype=np.float32)
        frame_acc = np.gradient(frame_omega, axis=0) / self.sample_interval

        return frame_acc


class ImageFeaturesDataset(BaseFeaturesDataset):
    def __init__(self, data_path, window_size=3, sample_frequency=10):
        super().__init__(data_path, window_size, sample_frequency)

    def __getitem__(self, idx):
        seq_subject, start_3d, end_3d = self.seq_pairs[idx]
        db = self.metadata[seq_subject]

        img_feats = []
        target = db['target']
        for d in db["data"][start_3d:end_3d]:
            img_feats.append(d['features'])

        img_feats = np.array(img_feats, dtype=np.float32)

        return img_feats, self.class_dict[target]
    

class ImputationDataset_IRV(PoseFeaturesDataset):
    '''
    IMPUTATION DATASET CLASS DEFINITION
        assumes datasets are saved in csv files with the form
        (features*,class,subj) or (features*,class)
    '''
    def __init__(self,
                 data_fp, 
                 sample_frequency, window_size=5,window_overlap=0.5,
                 mean_mask_length=3, masking_ratio=0.15, include_2d=False,include_3d=False,transform=None):

        super().__init__(data_fp, sample_frequency,
                         window_size,window_overlap,include_2d,include_3d,transform)

        self.lm = mean_mask_length
        self.r = masking_ratio

    def __getitem__(self, idx):
        pose,_ = PoseFeaturesDataset.__getitem__(idx)

        mask = noise_mask(pose_feats, lm=self.lm, r=self.r)
    
        if self.transform:
            pose_feats = self.transform(pose_feats)

        masked_ts = pose * mask
        target = np.deepcopy(pose)
        mask = mask.reshape(pose_feats.shape)

        return masked_ts,target, mask

