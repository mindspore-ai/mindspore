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
"""Eval"""
import os
import time
import datetime
import glob
import numpy as np
import mindspore.nn as nn

from mindspore import Tensor, context
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype

from src.utils.logging import get_logger
from src.vgg import vgg19
from src.dataset import vgg_create_dataset
from src.dataset import classification_dataset

from model_utils.moxing_adapter import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_rank_id, get_device_num

class ParameterReduce(nn.Cell):
    """ParameterReduce"""
    def __init__(self):
        super(ParameterReduce, self).__init__()
        self.cast = P.Cast()
        self.reduce = P.AllReduce()

    def construct(self, x):
        one = self.cast(F.scalar_to_array(1.0), mstype.float32)
        out = x * one
        ret = self.reduce(out)
        return ret


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
        if config.device_target == "GPU":
            init()
            device_id = get_rank()
            device_num = get_group_size()
        elif config.device_target == "Ascend":
            device_id = get_device_id()
            device_num = get_device_num()
        else:
            raise ValueError("Not support device_target.")

        # Each server contains 8 devices as most.
        if device_id % min(device_num, 8) == 0 and not os.path.exists(sync_lock):
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

        print("Device: {}, Finish sync unzip data from {} to {}.".format(device_id, zip_file_1, save_dir_1))

    config.log_path = os.path.join(config.output_path, config.log_path)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_eval():
    """run eval"""
    config.per_batch_size = config.batch_size
    config.image_size = list(map(int, config.image_size.split(',')))
    config.rank = get_rank_id()
    config.group_size = get_device_num()


    _enable_graph_kernel = config.device_target == "GPU"
    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=_enable_graph_kernel,
                        device_target=config.device_target, save_graphs=False)
    if os.getenv('DEVICE_ID', "not_set").isdigit() and config.device_target == "Ascend":
        context.set_context(device_id=int(os.getenv('DEVICE_ID')))

    config.outputs_dir = os.path.join(config.log_path,
                                      datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))

    config.logger = get_logger(config.outputs_dir, config.rank)
    config.logger.save_args(config)

    if config.dataset == "cifar10":
        net = vgg19(num_classes=config.num_classes, args=config)
        opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01, config.momentum,
                       weight_decay=config.weight_decay)
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'})

        param_dict = load_checkpoint(config.pre_trained)
        load_param_into_net(net, param_dict)
        net.set_train(False)
        dataset = vgg_create_dataset(config.data_dir, config.image_size, config.per_batch_size, training=False)
        res = model.eval(dataset)
        print("result: ", res)
    else:
        # network
        config.logger.important_info('start create network')
        if os.path.isdir(config.pre_trained):
            models = list(glob.glob(os.path.join(config.pre_trained, '*.ckpt')))
            print(models)
            if config.graph_ckpt:
                f = lambda x: -1 * int(os.path.splitext(os.path.split(x)[-1])[0].split('-')[-1].split('_')[0])
            else:
                f = lambda x: -1 * int(os.path.splitext(os.path.split(x)[-1])[0].split('_')[-1])
            config.models = sorted(models, key=f)
        else:
            config.models = [config.pre_trained,]

        for model in config.models:
            dataset = classification_dataset(config.data_dir, config.image_size, config.per_batch_size, mode='eval')
            eval_dataloader = dataset.create_tuple_iterator(output_numpy=True, num_epochs=1)
            network = vgg19(config.num_classes, config, phase="test")

            # pre_trained
            load_param_into_net(network, load_checkpoint(model))
            network.add_flags_recursive(fp16=True)

            img_tot = 0
            top1_correct = 0
            top5_correct = 0

            network.set_train(False)
            t_end = time.time()
            it = 0
            for data, gt_classes in eval_dataloader:
                output = network(Tensor(data, mstype.float32))
                output = output.asnumpy()

                top1_output = np.argmax(output, (-1))
                top5_output = np.argsort(output)[:, -5:]

                t1_correct = np.equal(top1_output, gt_classes).sum()
                top1_correct += t1_correct
                top5_correct += get_top5_acc(top5_output, gt_classes)
                img_tot += config.per_batch_size

                if config.rank == 0 and it == 0:
                    t_end = time.time()
                    it = 1
            if config.rank == 0:
                time_used = time.time() - t_end
                fps = (img_tot - config.per_batch_size) * config.group_size / time_used
                config.logger.info('Inference Performance: {:.2f} img/sec'.format(fps))
            results = [[top1_correct], [top5_correct], [img_tot]]
            config.logger.info('before results=%s', results)
            results = np.array(results)

            config.logger.info('after results=%s', results)
            top1_correct = results[0, 0]
            top5_correct = results[1, 0]
            img_tot = results[2, 0]
            acc1 = 100.0 * top1_correct / img_tot
            acc5 = 100.0 * top5_correct / img_tot
            config.logger.info('after allreduce eval: top1_correct={}, tot={},'
                               'acc={:.2f}%(TOP1)'.format(top1_correct, img_tot, acc1))
            config.logger.info('after allreduce eval: top5_correct={}, tot={},'
                               'acc={:.2f}%(TOP5)'.format(top5_correct, img_tot, acc5))


if __name__ == "__main__":
    run_eval()
