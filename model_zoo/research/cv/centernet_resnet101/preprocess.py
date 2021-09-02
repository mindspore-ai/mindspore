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
"""pre process for 310 inference"""
import os
import numpy as np
from src.model_utils.config import config, dataset_config, eval_config, net_config
from src.dataset import COCOHP


def preprocess(dataset_path, preprocess_path):
    """preprocess input images"""
    meta_path = os.path.join(preprocess_path, "meta/meta")
    result_path = os.path.join(preprocess_path, "data")
    if not os.path.exists(meta_path):
        os.makedirs(os.path.join(preprocess_path, "meta/meta"))
    if not os.path.exists(result_path):
        os.makedirs(os.path.join(preprocess_path, "data"))
    coco = COCOHP(dataset_config, run_mode="val", net_opt=net_config)
    coco.init(dataset_path, keep_res=False)
    dataset = coco.create_eval_dataset()
    name_list = []
    meta_list = []
    i = 0
    for data in dataset.create_dict_iterator(num_epochs=1):
        img_id = data['image_id'].asnumpy().reshape((-1))[0]
        image = data['image'].asnumpy()
        for scale in eval_config.multi_scales:
            image_preprocess, meta = coco.pre_process_for_test(image, img_id, scale)
        evl_file_name = "eval2017_image" + "_" + str(img_id) + ".bin"
        evl_file_path = result_path + "/" + evl_file_name
        image_preprocess.tofile(evl_file_path)
        meta_file_path = os.path.join(preprocess_path + "/meta/meta", str(img_id) + ".txt")
        with open(meta_file_path, 'w+') as f:
            f.write(str(meta))
        name_list.append(img_id)
        meta_list.append(meta)
        i += 1
        print(f"preprocess: no.[{i}], img_name:{img_id}")
    np.save(os.path.join(preprocess_path + "/meta", "name_list.npy"), np.array(name_list))
    np.save(os.path.join(preprocess_path + "/meta", "meta_list.npy"), np.array(meta_list))


if __name__ == '__main__':
    preprocess(config.val_data_dir, config.predict_dir)
