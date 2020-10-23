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
# ===========================================================================
"""DSCNN eval."""
import os
import datetime
import glob
import argparse

import numpy as np
from mindspore import context
from mindspore import Tensor, Model
from mindspore.common import dtype as mstype

from src.config import eval_config
from src.log import get_logger
from src.dataset import audio_dataset
from src.ds_cnn import DSCNN
from src.models import load_ckpt

def get_top5_acc(top5_arg, gt_class):
    sub_count = 0
    for top5, gt in zip(top5_arg, gt_class):
        if gt in top5:
            sub_count += 1
    return sub_count


def val(args, model, test_de):
    '''Eval.'''
    eval_dataloader = test_de.create_tuple_iterator()
    img_tot = 0
    top1_correct = 0
    top5_correct = 0
    for data, gt_classes in eval_dataloader:
        output = model.predict(Tensor(data, mstype.float32))
        output = output.asnumpy()
        top1_output = np.argmax(output, (-1))
        top5_output = np.argsort(output)[:, -5:]
        gt_classes = gt_classes.asnumpy()
        t1_correct = np.equal(top1_output, gt_classes).sum()
        top1_correct += t1_correct
        top5_correct += get_top5_acc(top5_output, gt_classes)
        img_tot += output.shape[0]

    results = [[top1_correct], [top5_correct], [img_tot]]

    results = np.array(results)

    top1_correct = results[0, 0]
    top5_correct = results[1, 0]
    img_tot = results[2, 0]
    acc1 = 100.0 * top1_correct / img_tot
    acc5 = 100.0 * top5_correct / img_tot
    if acc1 > args.best_acc:
        args.best_acc = acc1
        args.best_index = args.index
    args.logger.info('Eval: top1_cor:{}, top5_cor:{}, tot:{}, acc@1={:.2f}%, acc@5={:.2f}%' \
                     .format(top1_correct, top5_correct, img_tot, acc1, acc5))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', type=int, default=1, help='which device the model will be trained on')
    args, model_settings = eval_config(parser)
    context.set_context(mode=context.GRAPH_MODE, device_target="Davinci", device_id=args.device_id)

    # Logger
    args.outputs_dir = os.path.join(args.log_path, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    args.logger = get_logger(args.outputs_dir)
    # show args
    args.logger.save_args(args)
    # find model path
    if os.path.isdir(args.model_dir):
        models = list(glob.glob(os.path.join(args.model_dir, '*.ckpt')))
        print(models)
        f = lambda x: -1 * int(os.path.splitext(os.path.split(x)[-1])[0].split('-')[0].split('epoch')[-1])
        args.models = sorted(models, key=f)
    else:
        args.models = [args.model_dir]

    args.best_acc = 0
    args.index = 0
    args.best_index = 0
    for model_path in args.models:
        test_de = audio_dataset(args.feat_dir, 'testing', model_settings['spectrogram_length'],
                                model_settings['dct_coefficient_count'], args.per_batch_size)
        network = DSCNN(model_settings, args.model_size_info)

        load_ckpt(network, model_path, False)
        network.set_train(False)
        model = Model(network)
        args.logger.info('load model {} success'.format(model_path))
        val(args, model, test_de)
        args.index += 1

    args.logger.info('Best model:{} acc:{:.2f}%'.format(args.models[args.best_index], args.best_acc))

if __name__ == "__main__":
    main()
