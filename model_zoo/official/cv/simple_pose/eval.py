# Copyright 2020 Huawei Technologies Co., Ltd
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
import time
import numpy as np

from mindspore import Tensor, float32, context
from mindspore.common import set_seed
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.dataset import flip_pairs, keypoint_dataset
from src.evaluate.coco_eval import evaluate
from src.model import get_pose_net
from src.utils.transform import flip_back
from src.predict import get_final_preds
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.config import config
from src.model_utils.device_adapter import get_device_id

def validate(cfg, val_dataset, model, output_dir):
    # switch to evaluate mode
    model.set_train(False)

    # init record
    num_samples = val_dataset.get_dataset_size() * cfg.TEST.BATCH_SIZE
    all_preds = np.zeros((num_samples, cfg.MODEL.NUM_JOINTS, 3),
                         dtype=np.float32)
    all_boxes = np.zeros((num_samples, 2))
    image_id = []
    idx = 0

    # start eval
    start = time.time()
    for item in val_dataset.create_dict_iterator():
        # input data
        inputs = item['image'].asnumpy()
        # compute output
        output = model(Tensor(inputs, float32)).asnumpy()
        if cfg.TEST.FLIP_TEST:
            inputs_flipped = Tensor(inputs[:, :, :, ::-1], float32)
            output_flipped = model(inputs_flipped)
            output_flipped = flip_back(output_flipped.asnumpy(), flip_pairs)

            # feature is not aligned, shift flipped heatmap for higher accuracy
            if cfg.TEST.SHIFT_HEATMAP:
                output_flipped[:, :, :, 1:] = \
                    output_flipped.copy()[:, :, :, 0:-1]

            output = (output + output_flipped) * 0.5

        # meta data
        c = item['center'].asnumpy()
        s = item['scale'].asnumpy()
        score = item['score'].asnumpy()
        file_id = list(item['id'].asnumpy())

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
        if idx % 1024 == 0:
            print('{} samples validated in {} seconds'.format(idx, time.time() - start))
            start = time.time()

    print(all_preds[:idx].shape, all_boxes[:idx].shape, len(image_id))
    _, perf_indicator = evaluate(
        cfg, all_preds[:idx], output_dir, all_boxes[:idx], image_id)
    print("AP:", perf_indicator)
    return perf_indicator

def modelarts_pre_process():
    '''modelarts pre process function.'''
    config.TEST.MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.TEST.MODEL_FILE)
    config.DATASET.ROOT = config.data_path


@moxing_wrapper(pre_process=modelarts_pre_process)
def main():
    # init seed
    set_seed(1)

    # set context
    device_id = get_device_id()
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="Ascend", save_graphs=False, device_id=device_id)

    # init model
    model = get_pose_net(config, is_train=False)

    # load parameters
    ckpt_file = config.TEST.MODEL_FILE
    print('loading model ckpt from {}'.format(ckpt_file))
    load_param_into_net(model, load_checkpoint(ckpt_file))

    # Data loading code
    valid_dataset, _ = keypoint_dataset(
        config,
        bbox_file=config.TEST.COCO_BBOX_FILE,
        train_mode=False,
        num_parallel_workers=config.TEST.DATALOADER_WORKERS,
    )

    # evaluate on validation set
    output_dir = ckpt_file.split('.')[0]
    if config.enable_modelarts:
        output_dir = config.output_path
    validate(config, valid_dataset, model, output_dir)


if __name__ == '__main__':
    main()
