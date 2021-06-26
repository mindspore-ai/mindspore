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
# ===========================================================================
"""DSCNN eval."""
import os
import datetime
import glob
import numpy as np
from mindspore import context
from mindspore import Tensor, Model
from mindspore.common import dtype as mstype
from src.log import get_logger
from src.dataset import audio_dataset
from src.ds_cnn import DSCNN
from src.models import load_ckpt
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id


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


@moxing_wrapper(pre_process=None)
def main():
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=get_device_id())

    # Logger
    config.outputs_dir = os.path.join(config.log_path, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    config.logger = get_logger(config.outputs_dir)
    # show args
    config.logger.save_args(config)

    # find model path
    if os.path.isdir(config.model_dir):
        models = list(glob.glob(os.path.join(config.model_dir, '*.ckpt')))
        print(models)
        f = lambda x: -1 * int(os.path.splitext(os.path.split(x)[-1])[0].split('-')[0].split('epoch')[-1])
        config.models = sorted(models, key=f)
    else:
        config.models = [config.model_dir]

    config.best_acc = 0
    config.index = 0
    config.best_index = 0
    for model_path in config.models:
        test_de = audio_dataset(config.eval_feat_dir, 'testing', config.model_setting_spectrogram_length,
                                config.model_setting_dct_coefficient_count, config.per_batch_size)
        network = DSCNN(config, config.model_size_info)

        load_ckpt(network, model_path, False)
        network.set_train(False)
        model = Model(network)
        config.logger.info('load model %s success', model_path)
        val(config, model, test_de)
        config.index += 1

    config.logger.info('Best model:{} acc:{:.2f}%'.format(config.models[config.best_index], config.best_acc))


if __name__ == "__main__":
    main()
