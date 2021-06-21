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

from src.evaluate.coco_eval import evaluate
from src.utils.transform import flip_back
from src.predict import get_final_preds
from src.dataset import flip_pairs
from src.model_utils.config import config


def get_acc():
    '''calculate accuracy'''
    ckpt_file = config.TEST.MODEL_FILE
    output_dir = ckpt_file.split('.')[0]
    if config.enable_modelarts:
        output_dir = config.output_path
    cfg = config

    # init record
    file_num = len(os.listdir(config.INFER.POST_RESULT_PATH)) // 2
    num_samples = file_num * cfg.TEST.BATCH_SIZE
    all_preds = np.zeros((num_samples, cfg.MODEL.NUM_JOINTS, 3),
                         dtype=np.float32)
    all_boxes = np.zeros((num_samples, 2))
    image_id = []
    idx = 0
    bs = config.TEST.BATCH_SIZE
    h, w = config.POSE_RESNET.HEATMAP_SIZE[1], config.POSE_RESNET.HEATMAP_SIZE[0]
    shape = [bs, config.MODEL.NUM_JOINTS, h, w]

    for i in range(file_num):
        f = os.path.join(config.INFER.POST_RESULT_PATH, "sp_bs" + str(bs) + "_" + str(i) + "_0.bin")
        output = np.fromfile(f, np.float32).reshape(shape)
        if cfg.TEST.FLIP_TEST:
            f = os.path.join(config.INFER.POST_RESULT_PATH, "sp_flip_bs" + str(bs) + "_" + str(i) + "_0.bin")
            output_flipped = np.fromfile(f, np.float32).reshape(shape)
            output_flipped = flip_back(output_flipped, flip_pairs)

            # feature is not aligned, shift flipped heatmap for higher accuracy
            if cfg.TEST.SHIFT_HEATMAP:
                output_flipped[:, :, :, 1:] = \
                    output_flipped.copy()[:, :, :, 0:-1]

            output = (output + output_flipped) * 0.5

        # meta data
        center_path = os.path.join(config.INFER.PRE_RESULT_PATH, "center")
        scale_path = os.path.join(config.INFER.PRE_RESULT_PATH, "scale")
        score_path = os.path.join(config.INFER.PRE_RESULT_PATH, "score")
        id_path = os.path.join(config.INFER.PRE_RESULT_PATH, "id")
        file_name = "sp_bs" + str(bs) + "_" + str(i) + ".npy"
        c = np.load(os.path.join(center_path, file_name))
        s = np.load(os.path.join(scale_path, file_name))
        score = np.load(os.path.join(score_path, file_name))
        file_id = np.load(os.path.join(id_path, file_name))

        # pred by heatmaps
        preds, maxvals = get_final_preds(cfg, output.copy(), c, s)
        num_images, _ = preds.shape[:2]
        all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
        all_preds[idx:idx + num_images, :, 2:3] = maxvals
        # double check this all_boxes parts
        all_boxes[idx:idx + num_images, 0] = np.prod(s * 200, 1)
        all_boxes[idx:idx + num_images, 1] = score
        image_id.extend(file_id)
        idx += num_images


    print(all_preds[:idx].shape, all_boxes[:idx].shape, len(image_id))
    _, perf_indicator = evaluate(
        cfg, all_preds[:idx], output_dir, all_boxes[:idx], image_id)
    print("AP:", perf_indicator)

if __name__ == '__main__':
    get_acc()
