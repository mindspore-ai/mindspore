# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

"""
##############test densenet example#################
python eval.py --net densenet121 --dataset imagenet --data_dir /PATH/TO/DATASET --pretrained /PATH/TO/CHECKPOINT
"""

import os
import datetime
import glob
import numpy as np
from mindspore import context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.communication.management import init, get_group_size, release
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype
from src.utils.logging import get_logger
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.config import config
from src.model_utils.device_adapter import get_device_id, get_rank_id


class ParameterReduce(nn.Cell):
    """
    reduce parameter
    """
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


def generate_results(model, rank, group_size, top1_correct, top5_correct, img_tot):
    model_md5 = model.replace('/', '')
    tmp_dir = '../cache'
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    top1_correct_npy = '{}/top1_rank_{}_{}.npy'.format(tmp_dir, rank, model_md5)
    top5_correct_npy = '{}/top5_rank_{}_{}.npy'.format(tmp_dir, rank, model_md5)
    img_tot_npy = '{}/img_tot_rank_{}_{}.npy'.format(tmp_dir, rank, model_md5)
    np.save(top1_correct_npy, top1_correct)
    np.save(top5_correct_npy, top5_correct)
    np.save(img_tot_npy, img_tot)
    while True:
        rank_ok = True
        for other_rank in range(group_size):
            top1_correct_npy = '{}/top1_rank_{}_{}.npy'.format(tmp_dir, other_rank, model_md5)
            top5_correct_npy = '{}/top5_rank_{}_{}.npy'.format(tmp_dir, other_rank, model_md5)
            img_tot_npy = '{}/img_tot_rank_{}_{}.npy'.format(tmp_dir, other_rank, model_md5)
            if not os.path.exists(top1_correct_npy) or not os.path.exists(top5_correct_npy) \
               or not os.path.exists(img_tot_npy):
                rank_ok = False
        if rank_ok:
            break

    top1_correct_all = 0
    top5_correct_all = 0
    img_tot_all = 0
    for other_rank in range(group_size):
        top1_correct_npy = '{}/top1_rank_{}_{}.npy'.format(tmp_dir, other_rank, model_md5)
        top5_correct_npy = '{}/top5_rank_{}_{}.npy'.format(tmp_dir, other_rank, model_md5)
        img_tot_npy = '{}/img_tot_rank_{}_{}.npy'.format(tmp_dir, other_rank, model_md5)
        top1_correct_all += np.load(top1_correct_npy)
        top5_correct_all += np.load(top5_correct_npy)
        img_tot_all += np.load(img_tot_npy)
    return [[top1_correct_all], [top5_correct_all], [img_tot_all]]


@moxing_wrapper()
def test():
    """
    network eval function. Get top1 and top5 ACC from classification for imagenet,
    and top1 ACC for cifar10.
    The result will be save at [./outputs] by default.
    """
    config.image_size = list(map(int, config.image_size.split(',')))

    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target,
                        save_graphs=True)
    if config.device_target == 'Ascend':
        devid = get_device_id()
        context.set_context(device_id=devid)

    # init distributed
    if config.is_distributed:
        init()
        config.rank = get_rank_id()
        config.group_size = get_group_size()

    config.outputs_dir = os.path.join(config.log_path, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))

    config.logger = get_logger(config.outputs_dir, config.rank)
    config.logger.save_args(config)

    # network
    config.logger.important_info('start create network')
    if os.path.isdir(config.ckpt_files):
        models = list(glob.glob(os.path.join(config.ckpt_files, '*.ckpt')))

        f = lambda x: -1 * int(os.path.splitext(os.path.split(x)[-1])[0].split('-')[-1].split('_')[0])

        config.models = sorted(models, key=f)
    else:
        config.models = [config.ckpt_files,]

    if config.net == "densenet100":
        from src.network.densenet import DenseNet100 as DenseNet
    else:
        from src.network.densenet import DenseNet121 as DenseNet

    if config.dataset == "cifar10":
        from src.datasets import classification_dataset_cifar10 as classification_dataset
    else:
        from src.datasets import classification_dataset_imagenet as classification_dataset

    for model in config.models:
        de_dataset = classification_dataset(config.eval_data_dir, image_size=config.image_size,
                                            per_batch_size=config.per_batch_size,
                                            max_epoch=1, rank=config.rank, group_size=config.group_size,
                                            mode='eval')
        eval_dataloader = de_dataset.create_tuple_iterator()
        network = DenseNet(config.num_classes)

        param_dict = load_checkpoint(model)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('network.'):
                param_dict_new[key[8:]] = values
            else:
                param_dict_new[key] = values
        load_param_into_net(network, param_dict_new)
        config.logger.info('load model %s success', str(model))

        if config.device_target == 'Ascend':
            network.add_flags_recursive(fp16=True)

        img_tot = 0
        top1_correct = 0
        top5_correct = 0
        network.set_train(False)
        for data, gt_classes in eval_dataloader:
            output = network(Tensor(data, mstype.float32))
            output = output.asnumpy()
            gt_classes = gt_classes.asnumpy()

            top1_output = np.argmax(output, (-1))
            top5_output = np.argsort(output)[:, -5:]

            t1_correct = np.equal(top1_output, gt_classes).sum()
            top1_correct += t1_correct
            top5_correct += get_top5_acc(top5_output, gt_classes)
            img_tot += config.per_batch_size

        results = [[top1_correct], [top5_correct], [img_tot]]
        config.logger.info('before results=%s', str(results))
        if config.is_distributed:
            results = generate_results(model, config.rank, config.group_size, top1_correct,
                                       top5_correct, img_tot)
            results = np.array(results)
        else:
            results = np.array(results)

        config.logger.info('after results=%s', str(results))
        top1_correct = results[0, 0]
        top5_correct = results[1, 0]
        img_tot = results[2, 0]
        acc1 = 100.0 * top1_correct / img_tot
        acc5 = 100.0 * top5_correct / img_tot
        config.logger.info('after allreduce eval: top1_correct={}, tot={}, acc={:.2f}%'.format(top1_correct
                                                                                               , img_tot, acc1))
        if config.dataset == 'imagenet':
            config.logger.info('after allreduce eval: top5_correct={}, tot={}, acc={:.2f}%'.format(top5_correct
                                                                                                   , img_tot, acc5))
    if config.is_distributed:
        release()


if __name__ == "__main__":
    test()
