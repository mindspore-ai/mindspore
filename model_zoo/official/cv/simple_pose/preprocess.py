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
import os
import numpy as np

from src.dataset import keypoint_dataset
from src.model_utils.config import config

def get_bin():
    ''' get bin files'''
    valid_dataset, _ = keypoint_dataset(
        config,
        bbox_file=config.TEST.COCO_BBOX_FILE,
        train_mode=False,
        num_parallel_workers=config.TEST.DATALOADER_WORKERS,
    )
    inputs_path = os.path.join(config.INFER.PRE_RESULT_PATH, "00_data")
    os.makedirs(inputs_path)

    center_path = os.path.join(config.INFER.PRE_RESULT_PATH, "center")
    os.makedirs(center_path)

    scale_path = os.path.join(config.INFER.PRE_RESULT_PATH, "scale")
    os.makedirs(scale_path)

    score_path = os.path.join(config.INFER.PRE_RESULT_PATH, "score")
    os.makedirs(score_path)

    id_path = os.path.join(config.INFER.PRE_RESULT_PATH, "id")
    os.makedirs(id_path)

    for i, item in enumerate(valid_dataset.create_dict_iterator(output_numpy=True)):
        file_name = "sp_bs" + str(config.TEST.BATCH_SIZE) + "_" + str(i) + ".bin"
        # input data
        inputs = item['image']
        inputs_file_path = os.path.join(inputs_path, file_name)
        inputs.tofile(inputs_file_path)
        if config.TEST.FLIP_TEST:
            inputs_flipped = inputs[:, :, :, ::-1]
            file_name = "sp_flip_bs" + str(config.TEST.BATCH_SIZE) + "_" + str(i) + ".bin"
            inputs_file_path = os.path.join(inputs_path, file_name)
            inputs_flipped.tofile(inputs_file_path)
        file_name = "sp_bs" + str(config.TEST.BATCH_SIZE) + "_" + str(i) + ".npy"
        np.save(os.path.join(center_path, file_name), item['center'])
        np.save(os.path.join(scale_path, file_name), item['scale'])
        np.save(os.path.join(score_path, file_name), item['score'])
        np.save(os.path.join(id_path, file_name), item['id'])
    print("=" * 20, "export bin files finished", "=" * 20)


if __name__ == '__main__':
    get_bin()
