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
"""Face attribute eval."""
import os
import time
import numpy as np

from mindspore import context
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import dtype as mstype

from src.dataset_eval import data_generator_eval
from src.FaceAttribute.resnet18 import get_resnet18

from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_device_num


def softmax(x, axis=0):
    return np.exp(x) / np.sum(np.exp(x), axis=axis)


def load_pretrain(checkpoint, network):
    '''load pretrain model.'''
    if os.path.isfile(checkpoint):
        param_dict = load_checkpoint(checkpoint)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('network.'):
                param_dict_new[key[8:]] = values
            else:
                param_dict_new[key] = values
        load_param_into_net(network, param_dict_new)
        print('-----------------------load model success-----------------------')
    else:
        print('-----------------------load model failed-----------------------')
    return network


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


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_eval():
    '''run eval.'''
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False,
                        device_id=get_device_id())

    network = get_resnet18(config)
    ckpt_path = config.model_path
    network = load_pretrain(ckpt_path, network)

    network.set_train(False)

    de_dataloader, steps_per_epoch, _ = data_generator_eval(config)

    total_data_num_age, total_data_num_gen, total_data_num_mask = 0, 0, 0
    age_num, gen_num, mask_num = 0, 0, 0
    gen_tp_num, mask_tp_num, gen_fp_num = 0, 0, 0
    mask_fp_num, gen_fn_num, mask_fn_num = 0, 0, 0
    for step_i, (data, gt_classes) in enumerate(de_dataloader):

        print('evaluating {}/{} ...'.format(step_i + 1, steps_per_epoch))

        data_tensor = Tensor(data, dtype=mstype.float32)
        fea = network(data_tensor)

        gt_age, gt_gen, gt_mask = gt_classes[0]

        age_result, gen_result, mask_result = fea

        age_result_np = age_result.asnumpy()
        gen_result_np = gen_result.asnumpy()
        mask_result_np = mask_result.asnumpy()

        age_prob = softmax(age_result_np[0].astype(np.float32)).tolist()
        gen_prob = softmax(gen_result_np[0].astype(np.float32)).tolist()
        mask_prob = softmax(mask_result_np[0].astype(np.float32)).tolist()

        age = age_prob.index(max(age_prob))
        gen = gen_prob.index(max(gen_prob))
        mask = mask_prob.index(max(mask_prob))

        if gt_age == age:
            age_num += 1
        if gt_gen == gen:
            gen_num += 1
        if gt_mask == mask:
            mask_num += 1

        if gen == 1:
            if gt_gen == 1:
                gen_tp_num += 1
            elif gt_gen == 0:
                gen_fp_num += 1
        elif gen == 0 and gt_gen == 1:
            gen_fn_num += 1

        if gt_mask == 1 and mask == 1:
            mask_tp_num += 1
        if gt_mask == 0 and mask == 1:
            mask_fp_num += 1
        if gt_mask == 1 and mask == 0:
            mask_fn_num += 1

        if gt_age != -1:
            total_data_num_age += 1
        if gt_gen != -1:
            total_data_num_gen += 1
        if gt_mask != -1:
            total_data_num_mask += 1

    age_accuracy = float(age_num) / float(total_data_num_age)

    gen_precision = float(gen_tp_num) / (float(gen_tp_num) + float(gen_fp_num))
    gen_recall = float(gen_tp_num) / (float(gen_tp_num) + float(gen_fn_num))
    gen_accuracy = float(gen_num) / float(total_data_num_gen)
    gen_f1 = 2. * gen_precision * gen_recall / (gen_precision + gen_recall)

    mask_precision = float(mask_tp_num) / (float(mask_tp_num) + float(mask_fp_num))
    mask_recall = float(mask_tp_num) / (float(mask_tp_num) + float(mask_fn_num))
    mask_accuracy = float(mask_num) / float(total_data_num_mask)
    mask_f1 = 2. * mask_precision * mask_recall / (mask_precision + mask_recall)

    print('model: ', ckpt_path)
    print('total age num: ', total_data_num_age)
    print('total gen num: ', total_data_num_gen)
    print('total mask num: ', total_data_num_mask)
    print('age accuracy: ', age_accuracy)
    print('gen accuracy: ', gen_accuracy)
    print('mask accuracy: ', mask_accuracy)
    print('gen precision: ', gen_precision)
    print('gen recall: ', gen_recall)
    print('gen f1: ', gen_f1)
    print('mask precision: ', mask_precision)
    print('mask recall: ', mask_recall)
    print('mask f1: ', mask_f1)

    model_name = os.path.basename(ckpt_path).split('.')[0]
    model_dir = os.path.dirname(ckpt_path)
    result_txt = os.path.join(model_dir, model_name + '.txt')
    if os.path.exists(result_txt):
        os.remove(result_txt)
    with open(result_txt, 'a') as ft:
        ft.write('model: {}\n'.format(ckpt_path))
        ft.write('total age num: {}\n'.format(total_data_num_age))
        ft.write('total gen num: {}\n'.format(total_data_num_gen))
        ft.write('total mask num: {}\n'.format(total_data_num_mask))
        ft.write('age accuracy: {}\n'.format(age_accuracy))
        ft.write('gen accuracy: {}\n'.format(gen_accuracy))
        ft.write('mask accuracy: {}\n'.format(mask_accuracy))
        ft.write('gen precision: {}\n'.format(gen_precision))
        ft.write('gen recall: {}\n'.format(gen_recall))
        ft.write('gen f1: {}\n'.format(gen_f1))
        ft.write('mask precision: {}\n'.format(mask_precision))
        ft.write('mask recall: {}\n'.format(mask_recall))
        ft.write('mask f1: {}\n'.format(mask_f1))


if __name__ == '__main__':
    run_eval()
