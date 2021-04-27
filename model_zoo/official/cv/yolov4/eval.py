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
"""YoloV4 eval."""
import os
import argparse
import datetime
import time

from mindspore import Tensor
from mindspore.context import ParallelMode
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore as ms

from src.yolo import YOLOV4CspDarkNet53
from src.logger import get_logger
from src.yolo_dataset import create_yolo_dataset
from src.config import ConfigYOLOV4CspDarkNet53
from src.eval_utils import apply_eval

parser = argparse.ArgumentParser('mindspore coco testing')

# device related
parser.add_argument('--device_target', type=str, default='Ascend',
                    help='device where the code will be implemented. (Default: Ascend)')

# dataset related
parser.add_argument('--data_dir', type=str, default='', help='train data dir')
parser.add_argument('--per_batch_size', default=1, type=int, help='batch size for per gpu')

# network related
parser.add_argument('--pretrained', default='', type=str, help='model_path, local pretrained model to load')

# logging related
parser.add_argument('--log_path', type=str, default='outputs/', help='checkpoint save location')

# detect_related
parser.add_argument('--ann_val_file', type=str, default='', help='path to annotation')
parser.add_argument('--testing_shape', type=str, default='', help='shape for test ')

args, _ = parser.parse_known_args()

config = ConfigYOLOV4CspDarkNet53()
args.nms_thresh = config.nms_thresh
args.ignore_threshold = config.eval_ignore_threshold
args.data_root = os.path.join(args.data_dir, 'val2017')
args.ann_val_file = os.path.join(args.data_dir, 'annotations/instances_val2017.json')


def convert_testing_shape(args_testing_shape):
    """Convert testing shape to list."""
    testing_shape = [int(args_testing_shape), int(args_testing_shape)]
    return testing_shape


if __name__ == "__main__":
    start_time = time.time()
    device_id = int(os.getenv('DEVICE_ID')) if os.getenv('DEVICE_ID') else 0
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=device_id)

    # logger
    args.outputs_dir = os.path.join(args.log_path,
                                    datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    rank_id = int(os.environ.get('RANK_ID')) if os.environ.get('RANK_ID') else 0
    args.logger = get_logger(args.outputs_dir, rank_id)

    context.reset_auto_parallel_context()
    parallel_mode = ParallelMode.STAND_ALONE
    context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=1)

    args.logger.info('Creating Network....')
    network = YOLOV4CspDarkNet53()

    args.logger.info(args.pretrained)
    if os.path.isfile(args.pretrained):
        param_dict = load_checkpoint(args.pretrained)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('yolo_network.'):
                param_dict_new[key[13:]] = values
            else:
                param_dict_new[key] = values
        load_param_into_net(network, param_dict_new)
        args.logger.info('load_model {} success'.format(args.pretrained))
    else:
        args.logger.info('{} not exists or not a pre-trained file'.format(args.pretrained))
        assert FileNotFoundError('{} not exists or not a pre-trained file'.format(args.pretrained))
        exit(1)

    data_root = args.data_root
    ann_val_file = args.ann_val_file

    if args.testing_shape:
        config.test_img_shape = convert_testing_shape(args.testing_shape)

    ds, data_size = create_yolo_dataset(data_root, ann_val_file, is_training=False, batch_size=args.per_batch_size,
                                        max_epoch=1, device_num=1, rank=rank_id, shuffle=False,
                                        config=config)

    args.logger.info('testing shape : {}'.format(config.test_img_shape))
    args.logger.info('totol {} images to eval'.format(data_size))
    network.set_train(False)

    # init detection engine
    input_shape = Tensor(tuple(config.test_img_shape), ms.float32)
    args.logger.info('Start inference....')
    eval_param_dict = {"net": network, "dataset": ds, "data_size": data_size,
                       "anno_json": args.ann_val_file, "input_shape": input_shape, "args": args}
    eval_result, _ = apply_eval(eval_param_dict)

    cost_time = time.time() - start_time
    args.logger.info('\n=============coco eval reulst=========\n' + eval_result)
    args.logger.info('testing cost time {:.2f}h'.format(cost_time / 3600.))
