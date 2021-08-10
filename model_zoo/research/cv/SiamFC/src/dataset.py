# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""VID dataset"""
import os
import pickle
import hashlib
import cv2
import numpy as np
from src.config import config

class ImagnetVIDDataset():
    """
        used in GeneratorDataset to deal with image pair
        Args:
            db : lmdb file
            video_names : all video name
            data_dir : the location of image pair
            z_transforms : the transforms list used in exemplar
            x_transforms : the transforms list used in instance
            training : status of training
    """
    def __init__(self, db, video_names, data_dir, z_transforms, x_transforms, training=True):
        self.video_names = video_names
        self.data_dir = data_dir
        self.z_transforms = z_transforms
        self.x_transforms = x_transforms
        meta_data_path = os.path.join(data_dir, 'meta_data.pkl')
        self.meta_data = pickle.load(open(meta_data_path, 'rb'))
        self.meta_data = {x[0]: x[1] for x in self.meta_data}
        for key in self.meta_data.keys():
            trajs = self.meta_data[key]
            for trkid in list(trajs.keys()):
                if len(trajs[trkid]) < 2:
                    del trajs[trkid]
        self.txn = db.begin(write=False)
        self.num = len(self.video_names) if config.num_per_epoch is None or not \
                       training else config.num_per_epoch

    def imread(self, path):
        """
            read iamges according to path
            Args :
                path : the image path
        """
        key = hashlib.md5(path.encode()).digest()
        img_buffer = self.txn.get(key)
        img_buffer = np.frombuffer(img_buffer, np.uint8)
        img = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
        return img

    def _sample_weights(self, center, low_idx, high_idx, s_type='uniform'):
        """
           According to the center image to pick another image,setting the weights
           will be used in different type distribution
           Args:
                center : the position of center image
                low_idx : the  minimum of id
                high_idx : the max of id
                s_type : choose different distribution. "uniform", "sqrt", "linear"
                can be chosen

        """
        weights = list(range(low_idx, high_idx))
        weights.remove(center)
        weights = np.array(weights)
        if s_type == 'linear':
            weights = abs(weights - center)
        elif s_type == 'sqrt':
            weights = np.sqrt(abs(weights - center))
        elif s_type == 'uniform':
            weights = np.ones_like(weights)
        return weights / sum(weights)

    def __getitem__(self, idx):
        idx = idx % len(self.video_names)
        video = self.video_names[idx]
        trajs = self.meta_data[video]

        trkid = np.random.choice(list(trajs.keys()))
        traj = trajs[trkid]

        assert len(traj) > 1, "video_name: {}".format(video)
        exemplar_idx = np.random.choice(list(range(len(traj))))
        exemplar_name = os.path.join(self.data_dir, video,
                                     traj[exemplar_idx] + ".{:02d}.x.jpg".format(trkid))
        exemplar_img = self.imread(exemplar_name)
        exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_BGR2RGB)
        # sample instance
        low_idx = max(0, exemplar_idx - config.frame_range)
        up_idx = min(len(traj), exemplar_idx + config.frame_range)
        weights = self._sample_weights(exemplar_idx, low_idx, up_idx, config.sample_type)
        instance = np.random.choice(traj[low_idx:exemplar_idx] + traj[exemplar_idx + 1:up_idx],
                                    p=weights)
        instance_name = os.path.join(self.data_dir, video, instance + ".{:02d}.x.jpg".format(trkid))
        instance_img = self.imread(instance_name)
        instance_img = cv2.cvtColor(instance_img, cv2.COLOR_BGR2RGB)
        if np.random.rand(1) < config.gray_ratio:
            exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_RGB2GRAY)
            exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_GRAY2RGB)
            instance_img = cv2.cvtColor(instance_img, cv2.COLOR_RGB2GRAY)
            instance_img = cv2.cvtColor(instance_img, cv2.COLOR_GRAY2RGB)
        exemplar_img = self.z_transforms(exemplar_img)
        instance_img = self.x_transforms(instance_img)
        return exemplar_img, instance_img

    def __len__(self):
        return self.num
