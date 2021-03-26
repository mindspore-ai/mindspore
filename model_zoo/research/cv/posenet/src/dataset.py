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
"""Data operations, will be used in train.py and eval.py"""
import os
import numpy as np
from mindspore.mindrecord import FileWriter
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C

class Dataset:
    """dataset read"""
    def __init__(self, root, is_training=True):
        self.root = os.path.expanduser(root)
        self.train = is_training
        self.image_poses = []
        self.image_paths = []
        if self.train:
            txt_file = self.root + 'dataset_train.txt'
        else:
            txt_file = self.root + 'dataset_test.txt'

        with open(txt_file, 'r') as f:
            next(f)
            next(f)
            next(f)
            for line in f:
                fname, p0, p1, p2, p3, p4, p5, p6 = line.split()
                p0 = float(p0)
                p1 = float(p1)
                p2 = float(p2)
                p3 = float(p3)
                p4 = float(p4)
                p5 = float(p5)
                p6 = float(p6)
                self.image_poses.append((p0, p1, p2, p3, p4, p5, p6))
                self.image_paths.append(self.root + fname)

    def __getdata__(self):
        img_paths = self.image_paths
        img_poses = self.image_poses
        return img_paths, img_poses

    def __len__(self):
        return len(self.image_paths)

def data_to_mindrecord(data_path, is_training, mindrecord_file, file_num=1):
    """Create MindRecord file."""
    writer = FileWriter(mindrecord_file, file_num)
    data = Dataset(data_path, is_training)
    image_paths, image_poses = data.__getdata__()
    posenet_json = {
        "image": {"type": "bytes"},
        "image_pose": {"type": "float32", "shape": [-1]}
    }
    writer.add_schema(posenet_json, "posenet_json")

    image_files_num = len(image_paths)
    for ind, image_name in enumerate(image_paths):
        with open(image_name, 'rb') as f:
            image = f.read()
        image_pose = np.array(image_poses[ind])
        row = {"image": image, "image_pose": image_pose}
        if (ind + 1) % 10 == 0:
            print("writing {}/{} into mindrecord".format(ind + 1, image_files_num))
        writer.write_raw_data([row])
    writer.commit()

def create_posenet_dataset(mindrecord_file, batch_size=1, device_num=1, is_training=True, rank_id=0):
    """Create PoseNet dataset with MindDataset."""
    dataset = ds.MindDataset(mindrecord_file, columns_list=["image", "image_pose"],
                             num_shards=device_num, shard_id=rank_id,
                             num_parallel_workers=8, shuffle=True)

    decode = C.Decode()
    dataset = dataset.map(operations=decode, input_columns=["image"])
    transforms_list = []
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    if is_training:
        resized_op = C.Resize([455, 256])
        random_crop_op = C.RandomCrop(224)
        normlize_op = C.Normalize(mean=mean, std=std)
        to_tensor_op = C.HWC2CHW()
        transforms_list = [resized_op, random_crop_op, normlize_op, to_tensor_op]
    else:
        resized_op = C.Resize([455, 224])
        center_crop_op = C.CenterCrop(224)
        normlize_op = C.Normalize(mean=mean, std=std)
        to_tensor_op = C.HWC2CHW()
        transforms_list = [resized_op, center_crop_op, normlize_op, to_tensor_op]

    dataset = dataset.map(operations=transforms_list, input_columns=['image'])
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset
