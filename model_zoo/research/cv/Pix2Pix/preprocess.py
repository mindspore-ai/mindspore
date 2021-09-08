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
"""
    preprocess
"""
import os
import numpy as np
from src.dataset.pix2pix_dataset import pix2pixDataset_val, create_val_dataset
from src.utils.config import get_args

if __name__ == '__main__':
    args = get_args()
    result_path = "./preprocess_Result/"
    dataset_val = pix2pixDataset_val(root_dir=args.val_data_dir)
    ds_val = create_val_dataset(dataset_val)
    img_path = os.path.join(result_path, "00_data")
    os.makedirs(img_path)
    target_images_list = []
    for i, data in enumerate(ds_val.create_dict_iterator(output_numpy=True)):
        file_name = "Pix2Pix_data_bs" + str(args.batch_size) + "_" + str(i) + ".bin"
        file_path = img_path + "/" + file_name
        data['input_images'].tofile(file_path)
        target_images_list.append(data['target_images'])

    np.save(result_path + "target_images_ids.npy", target_images_list)
    print("=" * 20, "export bin files finished", "=" * 20)
