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
"""YoloV3 postprocess."""
import os
import datetime
import numpy as np
from PIL import Image
from eval import DetectionEngine
from model_utils.config import config

def get_img_size(file_name):
    img = Image.open(file_name)
    return img.size

if __name__ == "__main__":
    config.outputs_dir = os.path.join(config.log_path,
                                      datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    if not os.path.exists(config.outputs_dir):
        os.makedirs(config.outputs_dir)

    detection = DetectionEngine(config)
    bs = config.per_batch_size

    f_list = os.listdir(config.img_path)
    for f in f_list:
        image_size = get_img_size(os.path.join(config.img_path, f))
        f = f.split('.')[0]
        output_big = np.fromfile(os.path.join(config.result_path, f + '_0.bin'), np.float32).reshape(bs, 13, 13, 3, 85)
        output_me = np.fromfile(os.path.join(config.result_path, f + '_1.bin'), np.float32).reshape(bs, 26, 26, 3, 85)
        output_small = np.fromfile(os.path.join(config.result_path,
                                                f + '_2.bin'), np.float32).reshape(bs, 52, 52, 3, 85)
        image_id = [int(f.split('_')[-1])]
        image_shape = [[image_size[0], image_size[1]]]

        detection.detect([output_small, output_me, output_big], bs, image_shape, image_id)

    detection.do_nms_for_results()
    result_file_path = detection.write_result()
    eval_result = detection.get_eval_result()

    print('\n=============coco eval result=========\n' + eval_result)
