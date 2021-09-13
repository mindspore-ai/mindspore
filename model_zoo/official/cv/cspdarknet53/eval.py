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
"""evaluate imagenet."""
import os
import time
import datetime
import numpy as np

from mindspore import Tensor, context
from mindspore.common import dtype as mstype

from src.utils.logging import get_logger
from src.utils.auto_mixed_precision import auto_mixed_precision
from src.utils.var_init import load_pretrain_model
from src.image_classification import CSPDarknet53
from src.dataset import create_dataset

from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_rank_id, get_device_num

def get_top5_acc(top5_arg, gt_class):
    sub_count = 0
    for top5, gt in zip(top5_arg, gt_class):
        if gt in top5:
            sub_count += 1
    return sub_count


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
    '''Eval.'''
    config.image_size = list(map(int, config.image_size.split(',')))
    config.rank = get_rank_id()
    config.group_size = get_device_num()
    if config.is_distributed or config.group_size > 1:
        raise ValueError("Not support distribute eval.")
    config.outputs_dir = os.path.join(config.log_path,
                                      datetime.datetime.now().strftime("%Y-%m-%d_time_%H_%M_%S"))
    config.logger = get_logger(config.outputs_dir, config.rank)

    context.set_context(mode=context.GRAPH_MODE,
                        device_target=config.device_target, save_graphs=False, device_id=get_device_id())
    config.logger.save_args(config)

    # network
    config.logger.important_info('start create network')
    de_dataset = create_dataset(config.data_dir, config.image_size, config.per_batch_size,
                                config.rank, config.group_size, mode="eval")
    eval_dataloader = de_dataset.create_tuple_iterator(output_numpy=True, num_epochs=1)
    network = CSPDarknet53(num_classes=config.num_classes)
    load_pretrain_model(config.pretrained, network, config)

    img_tot = 0
    top1_correct = 0
    top5_correct = 0
    if config.device_target == "Ascend":
        network.to_float(mstype.float16)
    elif config.device_target == "GPU":
        auto_mixed_precision(network)
    else:
        raise ValueError("Not support device type: {}".format(config.device_target))
    network.set_train(False)
    t_start = time.time()
    for data, gt_classes in eval_dataloader:
        out = network(Tensor(data, mstype.float32))
        out = out.asnumpy()

        top1_output = np.argmax(out, (-1))
        top5_output = np.argsort(out)[:, -5:]

        t1_correct = np.equal(top1_output, gt_classes).sum()
        top1_correct += t1_correct
        top5_correct += get_top5_acc(top5_output, gt_classes)
        img_tot += config.per_batch_size

    t_end = time.time()
    if config.rank == 0:
        time_cost = t_end - t_start
        fps = (img_tot - config.per_batch_size) * config.group_size / time_cost
        config.logger.info('Inference Performance: {:.2f} img/sec'.format(fps))
    top1_acc = 100.0 * top1_correct / img_tot
    top5_acc = 100.0 * top5_correct / img_tot
    config.logger.info("top1_correct={}, tot={}, acc={:.2f}%(TOP1)".format(top1_correct, img_tot, top1_acc))
    config.logger.info("top5_correct={}, tot={}, acc={:.2f}%(TOP5)".format(top5_correct, img_tot, top5_acc))



if __name__ == '__main__':
    run_eval()
