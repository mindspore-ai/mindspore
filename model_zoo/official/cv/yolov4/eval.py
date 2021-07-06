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
import datetime
import time

from mindspore.context import ParallelMode
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.yolo import YOLOV4CspDarkNet53
from src.logger import get_logger
from src.yolo_dataset import create_yolo_dataset
from src.eval_utils import apply_eval

from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_device_num

config.data_root = os.path.join(config.data_dir, 'val2017')
config.ann_val_file = os.path.join(config.data_dir, 'annotations/instances_val2017.json')

def modelarts_pre_process():
    '''modelarts pre process function.'''
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, config.modelarts_dataset_unzip_name)):
            zip_isexist = zipfile.is_zipfile(zip_file)
            if zip_isexist:
                fz = zipfile.ZipFile(zip_file, 'r')
                data_num = len(fz.namelist())
                print("Extract Start...")
                print("unzip file num: {}".format(data_num))
                data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for file in fz.namelist():
                    if i % data_print == 0:
                        print("unzip percent: {}%".format(int(i * 100 / data_num)), flush=True)
                    i += 1
                    fz.extract(file, save_dir)
                print("cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),
                                                     int(int(time.time() - s_time) % 60)))
                print("Extract Done.")
            else:
                print("This is not zip.")
        else:
            print("Zip has been extracted.")

    if config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(config.data_path, config.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(config.data_path)

        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most.
        if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
            print("Zip file path: ", zip_file_1)
            print("Unzip file save dir: ", save_dir_1)
            unzip(zip_file_1, save_dir_1)
            print("===Finish extract data synchronization===")
            try:
                os.mknod(sync_lock)
            except IOError:
                pass

        while True:
            if os.path.exists(sync_lock):
                break
            time.sleep(1)

        print("Device: {}, Finish sync unzip data from {} to {}.".format(get_device_id(), zip_file_1, save_dir_1))

    config.log_path = os.path.join(config.output_path, config.log_path)

@moxing_wrapper(pre_process=modelarts_pre_process)
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
    network = YOLOV4CspDarkNet53()

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
                                        max_epoch=1, device_num=1, rank=rank_id, shuffle=False, default_config=config)

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
