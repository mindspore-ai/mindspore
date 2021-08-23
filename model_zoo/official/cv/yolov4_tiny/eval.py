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
"""YoloV4 eval."""
import os
import datetime
import time

from mindspore.context import ParallelMode
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.yolo import YOLOV4TinyCspDarkNet53
from src.logger import get_logger
from src.yolo_dataset import create_yolo_dataset
from src.eval_utils import apply_eval

from model_utils.config import config

config.data_root = os.path.join(config.data_dir, config.val_img_dir)
config.ann_val_file = os.path.join(config.data_dir, config.val_json_file)


def run_eval():
    start_time = time.time()
    device_id = int(os.getenv('DEVICE_ID')) if os.getenv('DEVICE_ID') else 0
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=device_id)

    # logger
    config.outputs_dir = os.path.join(config.log_path,
                                      datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    rank_id = int(os.environ.get('RANK_ID')) if os.environ.get('RANK_ID') else 0
    config.logger = get_logger(config.outputs_dir, rank_id)

    context.reset_auto_parallel_context()
    parallel_mode = ParallelMode.STAND_ALONE
    context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=1)

    config.logger.info('Creating Network....')
    network = YOLOV4TinyCspDarkNet53()

    config.logger.info(config.pretrained)
    if os.path.isfile(config.pretrained):
        param_dict = load_checkpoint(config.pretrained)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('yolo_network.'):
                param_dict_new[key[13:]] = values
            else:
                param_dict_new[key] = values
        load_param_into_net(network, param_dict_new)
        config.logger.info('load_model %s success', config.pretrained)
    else:
        config.logger.info('%s not exists or not a pre-trained file', config.pretrained)
        assert FileNotFoundError('{} not exists or not a pre-trained file'.format(config.pretrained))
        exit(1)

    data_root = config.data_root
    ann_val_file = config.ann_val_file

    ds, data_size = create_yolo_dataset(data_root, ann_val_file, is_training=False, batch_size=config.per_batch_size,
                                        max_epoch=1, device_num=1, rank=rank_id, shuffle=False,
                                        config=config)

    config.logger.info('testing shape : %s', config.test_img_shape)
    config.logger.info('totol %d images to eval', data_size)
    network.set_train(False)

    # init detection engine
    config.logger.info('Start inference....')
    eval_param_dict = {"net": network, "dataset": ds, "data_size": data_size,
                       "anno_json": config.ann_val_file, "args": config}
    eval_result, _ = apply_eval(eval_param_dict)

    cost_time = time.time() - start_time
    eval_log_string = '\n=============coco eval reulst=========\n' + eval_result
    config.logger.info(eval_log_string)
    config.logger.info('testing cost time %.2f h', cost_time / 3600.)


if __name__ == "__main__":
    run_eval()
