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
'''
This file evaluates the model used.
'''
from __future__ import division

import argparse
import os
import time
import numpy as np

from mindspore import Tensor, float32, context
from mindspore.common import set_seed
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.config import config
from src.pose_resnet import GetPoseResNet
from src.dataset import flip_pairs
from src.dataset import CreateDatasetCoco
from src.utils.coco import evaluate
from src.utils.transforms import flip_back
from src.utils.inference import get_final_preds

if config.MODELARTS.IS_MODEL_ARTS:
    import moxing as mox

set_seed(config.GENERAL.EVAL_SEED)
device_id = int(os.getenv('DEVICE_ID'))

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate')
    parser.add_argument('--data_url', required=True, default=None, help='Location of data.')
    parser.add_argument('--train_url', required=True, default=None, help='Location of evaluate outputs.')
    args = parser.parse_args()
    return args

def validate(cfg, val_dataset, model, output_dir, ann_path):
    '''
    validate
    '''
    model.set_train(False)
    num_samples = val_dataset.get_dataset_size() * cfg.TEST.BATCH_SIZE
    all_preds = np.zeros((num_samples, cfg.MODEL.NUM_JOINTS, 3),
                         dtype=np.float32)
    all_boxes = np.zeros((num_samples, 2))
    image_id = []
    idx = 0

    start = time.time()
    for item in val_dataset.create_dict_iterator():
        inputs = item['image'].asnumpy()
        output = model(Tensor(inputs, float32)).asnumpy()
        if cfg.TEST.FLIP_TEST:
            inputs_flipped = Tensor(inputs[:, :, :, ::-1], float32)
            output_flipped = model(inputs_flipped)
            output_flipped = flip_back(output_flipped.asnumpy(), flip_pairs)

            if cfg.TEST.SHIFT_HEATMAP:
                output_flipped[:, :, :, 1:] = \
                    output_flipped.copy()[:, :, :, 0:-1]

            output = (output + output_flipped) * 0.5

        c = item['center'].asnumpy()
        s = item['scale'].asnumpy()
        score = item['score'].asnumpy()
        file_id = list(item['id'].asnumpy())

        preds, maxvals = get_final_preds(cfg, output.copy(), c, s)
        num_images, _ = preds.shape[:2]
        all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
        all_preds[idx:idx + num_images, :, 2:3] = maxvals
        all_boxes[idx:idx + num_images, 0] = np.prod(s * 200, 1)
        all_boxes[idx:idx + num_images, 1] = score
        image_id.extend(file_id)
        idx += num_images
        if idx % 1024 == 0:
            print('{} samples validated in {} seconds'.format(idx, time.time() - start))
            start = time.time()

    print(all_preds[:idx].shape, all_boxes[:idx].shape, len(image_id))
    _, perf_indicator = evaluate(cfg, all_preds[:idx], output_dir, all_boxes[:idx], image_id, ann_path)
    print("AP:", perf_indicator)
    return perf_indicator


def main():
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="Ascend", save_graphs=False, device_id=device_id)

    args = parse_args()

    if config.MODELARTS.IS_MODEL_ARTS:
        mox.file.copy_parallel(src_url=args.data_url, dst_url=config.MODELARTS.CACHE_INPUT)

    model = GetPoseResNet(config)

    ckpt_name = ''
    if config.MODELARTS.IS_MODEL_ARTS:
        ckpt_name = config.MODELARTS.CACHE_INPUT
    else:
        ckpt_name = config.DATASET.ROOT
    ckpt_name = ckpt_name + config.TEST.MODEL_FILE
    print('loading model ckpt from {}'.format(ckpt_name))
    load_param_into_net(model, load_checkpoint(ckpt_name))

    valid_dataset = CreateDatasetCoco(
        train_mode=False,
        num_parallel_workers=config.TEST.NUM_PARALLEL_WORKERS,
    )

    ckpt_name = ckpt_name.split('/')
    ckpt_name = ckpt_name[len(ckpt_name) - 1]
    ckpt_name = ckpt_name.split('.')[0]
    output_dir = ''
    ann_path = ''
    if config.MODELARTS.IS_MODEL_ARTS:
        output_dir = config.MODELARTS.CACHE_OUTPUT
        ann_path = config.MODELARTS.CACHE_INPUT
    else:
        output_dir = config.TEST.OUTPUT_DIR
        ann_path = config.DATASET.ROOT
    output_dir = output_dir + ckpt_name
    ann_path = ann_path + config.DATASET.TEST_JSON
    validate(config, valid_dataset, model, output_dir, ann_path)

    if config.MODELARTS.IS_MODEL_ARTS:
        mox.file.copy_parallel(src_url=config.MODELARTS.CACHE_OUTPUT, dst_url=args.train_url)

if __name__ == '__main__':
    main()
