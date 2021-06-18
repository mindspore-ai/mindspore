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
"""preprocess"""
import os

import mindspore.dataset as de
import mindspore.dataset.vision.py_transforms as F
import mindspore.dataset.transforms.py_transforms as F2

from model_utils.config import config

def eval_data_generator(args):
    '''Build eval dataloader.'''
    mindrecord_path = args.mindrecord_path
    dst_w = args.dst_w
    dst_h = args.dst_h
    batch_size = 1
    #attri_num = args.attri_num
    transform_img = F2.Compose([F.Decode(),
                                F.Resize((dst_w, dst_h)),
                                F.ToTensor(),
                                F.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    de_dataset = de.MindDataset(mindrecord_path + "0", columns_list=["image", "label"])
    de_dataset = de_dataset.map(input_columns="image", operations=transform_img, num_parallel_workers=args.workers,
                                python_multiprocessing=True)
    de_dataset = de_dataset.batch(batch_size)

    #de_dataloader = de_dataset.create_tuple_iterator(output_numpy=True)
    steps_per_epoch = de_dataset.get_dataset_size()
    print("image number:{0}".format(steps_per_epoch))
    #num_classes = attri_num
    return de_dataset

if __name__ == "__main__":
    ds = eval_data_generator(config)
    cur_dir = os.getcwd()
    image_path = os.path.join(cur_dir, './data/image')
    if not os.path.isdir(image_path):
        os.makedirs(image_path)
    image_label_path = os.path.join(cur_dir, './data/label')
    if not os.path.isdir(image_label_path):
        os.makedirs(image_label_path)
        total = ds.get_dataset_size()
    iter_num = 0
    for data in ds.create_dict_iterator(output_numpy=True, num_epochs=1):
        file_name = "face_" + str(iter_num) + '.bin'
        img_np = data['image']
        image_label = data['label']
        img_np.tofile(os.path.join(image_path, file_name))
        image_label.tofile(os.path.join(image_label_path, file_name))
        iter_num += 1
    print("total num of images:", total)
