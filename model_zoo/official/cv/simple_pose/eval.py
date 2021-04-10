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
import argparse
import os
import time
import numpy as np


from mindspore import Tensor, float32, context
from mindspore.common import set_seed
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.config import config
from src.dataset import flip_pairs, keypoint_dataset
from src.evaluate.coco_eval import evaluate
from src.model import get_pose_net
from src.utils.transform import flip_back
from src.predict import get_final_preds


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument("--train_url", type=str, default="", help="")
    parser.add_argument("--data_url", type=str, default="", help="data")
    # output
    parser.add_argument('--output-url',
                        help='output dir',
                        type=str)
    # training
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        default=8,
                        type=int)
    parser.add_argument('--model-file',
                        help='model state file',
                        type=str)
    parser.add_argument('--use-detect-bbox',
                        help='use detect bbox',
                        action='store_true')
    parser.add_argument('--flip-test',
                        help='use flip test',
                        default=True,
                        action='store_true')
    parser.add_argument('--post-process',
                        help='use post process',
                        action='store_true')
    parser.add_argument('--shift-heatmap',
                        help='shift heatmap',
                        action='store_true')
    parser.add_argument('--coco-bbox-file',
                        help='coco detection bbox file',
                        type=str)

    args = parser.parse_args()

    return args


def reset_config(cfg, args):
    if args.use_detect_bbox:
        cfg.TEST.USE_GT_BBOX = not args.use_detect_bbox
    if args.flip_test:
        cfg.TEST.FLIP_TEST = args.flip_test
        print('use flip test:', cfg.TEST.FLIP_TEST)
    if args.post_process:
        cfg.TEST.POST_PROCESS = args.post_process
    if args.shift_heatmap:
        cfg.TEST.SHIFT_HEATMAP = args.shift_heatmap
    if args.model_file:
        cfg.TEST.MODEL_FILE = args.model_file
    if args.coco_bbox_file:
        cfg.TEST.COCO_BBOX_FILE = args.coco_bbox_file


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


def main():
    # init seed
    set_seed(1)

    # set context
    device_id = int(os.getenv('DEVICE_ID'))
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="Ascend", save_graphs=False, device_id=device_id)

    args = parse_args()
    # update config
    reset_config(config, args)

    # init model
    model = get_pose_net(config, is_train=False)

    # load parameters
    ckpt_name = config.TEST.MODEL_FILE
    print('loading model ckpt from {}'.format(ckpt_name))
    load_param_into_net(model, load_checkpoint(ckpt_name))

    # Data loading code
    valid_dataset, _ = keypoint_dataset(
        config,
        bbox_file=config.TEST.COCO_BBOX_FILE,
        train_mode=False,
        num_parallel_workers=args.workers,
    )

    # evaluate on validation set
    validate(config, valid_dataset, model, ckpt_name.split('.')[0])


if __name__ == '__main__':
    main()
