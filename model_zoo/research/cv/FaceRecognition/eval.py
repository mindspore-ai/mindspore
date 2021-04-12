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
"""Face Recognition eval."""
import os
import time
import math
from pprint import pformat
import numpy as np
import cv2

import mindspore.dataset.transforms.py_transforms as transforms
import mindspore.dataset.vision.py_transforms as vision
import mindspore.dataset as de
from mindspore import Tensor, context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.config import config_inference
from src.backbone.resnet import get_backbone
from src.my_logging import get_logger

devid = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=devid)


class TxtDataset():
    '''TxtDataset'''
    def __init__(self, root_all, filenames):
        super(TxtDataset, self).__init__()
        self.imgs = []
        self.labels = []
        for root, filename in zip(root_all, filenames):
            fin = open(filename, "r")
            for line in fin:
                self.imgs.append(os.path.join(root, line.strip().split(" ")[0]))
                self.labels.append(line.strip())
            fin.close()

    def __getitem__(self, index):
        try:
            img = cv2.cvtColor(cv2.imread(self.imgs[index]), cv2.COLOR_BGR2RGB)
        except:
            print(self.imgs[index])
            raise
        return img, index

    def __len__(self):
        return len(self.imgs)

    def get_all_labels(self):
        return self.labels

class DistributedSampler():
    '''DistributedSampler'''
    def __init__(self, dataset):
        self.dataset = dataset
        self.num_replicas = 1
        self.rank = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        indices = indices[self.rank::self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples

def get_dataloader(img_predix_all, img_list_all, batch_size, img_transforms):
    dataset = TxtDataset(img_predix_all, img_list_all)
    sampler = DistributedSampler(dataset)
    dataset_column_names = ["image", "index"]
    ds = de.GeneratorDataset(dataset, column_names=dataset_column_names, sampler=sampler)
    ds = ds.map(input_columns=["image"], operations=img_transforms)
    ds = ds.batch(batch_size, num_parallel_workers=8, drop_remainder=False)
    ds = ds.repeat(1)

    return ds, len(dataset), dataset.get_all_labels()

def generate_test_pair(jk_list, zj_list):
    '''generate_test_pair'''
    file_paths = [jk_list, zj_list]
    jk_dict = {}
    zj_dict = {}
    jk_zj_dict_list = [jk_dict, zj_dict]
    for path, x_dict in zip(file_paths, jk_zj_dict_list):
        with open(path, 'r') as fr:
            for line in fr:
                label = line.strip().split(' ')[1]
                tmp = x_dict.get(label, [])
                tmp.append(line.strip())
                x_dict[label] = tmp
    zj2jk_pairs = []
    for key in jk_dict:
        jk_file_list = jk_dict[key]
        zj_file_list = zj_dict[key]
        for zj_file in zj_file_list:
            zj2jk_pairs.append([zj_file, jk_file_list])
    return zj2jk_pairs

def check_minmax(args, data, min_value=0.99, max_value=1.01):
    min_data = data.min()
    max_data = data.max()
    if np.isnan(min_data) or np.isnan(max_data):
        args.logger.info('ERROR, nan happened, please check if used fp16 or other error')
        raise Exception
    if min_data < min_value or max_data > max_value:
        args.logger.info('ERROR, min or max is out if range, range=[{}, {}], minmax=[{}, {}]'.format(
            min_value, max_value, min_data, max_data))
        raise Exception

def get_model(args):
    '''get_model'''
    net = get_backbone(args)
    if args.fp16:
        net.add_flags_recursive(fp16=True)

    if args.weight.endswith('.ckpt'):
        param_dict = load_checkpoint(args.weight)
        param_dict_new = {}
        for key, value in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('network.'):
                param_dict_new[key[8:]] = value
            else:
                param_dict_new[key] = value
        load_param_into_net(net, param_dict_new)
        args.logger.info('INFO, ------------- load model success--------------')
    else:
        args.logger.info('ERROR, not support file:{}, please check weight in config.py'.format(args.weight))
        return 0
    net.set_train(False)
    return net

def topk(matrix, k, axis=1):
    '''topk'''
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        topk_index = np.argpartition(-matrix, k, axis=axis)[0:k, :]
        topk_data = matrix[topk_index, row_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[topk_index_sort, row_index]
        topk_index_sort = topk_index[0:k, :][topk_index_sort, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        topk_index = np.argpartition(-matrix, k, axis=axis)[:, 0:k]
        topk_data = matrix[column_index, topk_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[column_index, topk_index_sort]
        topk_index_sort = topk_index[:, 0:k][column_index, topk_index_sort]
    return topk_data_sort, topk_index_sort

def cal_topk(args, idx, zj2jk_pairs, test_embedding_tot, dis_embedding_tot):
    '''cal_topk'''
    args.logger.info('start idx:{} subprocess...'.format(idx))
    correct = np.array([0] * 2)
    tot = np.array([0])

    zj, jk_all = zj2jk_pairs[idx]
    zj_embedding = test_embedding_tot[zj]
    jk_all_embedding = np.concatenate([np.expand_dims(test_embedding_tot[jk], axis=0) for jk in jk_all], axis=0)
    args.logger.info('INFO, calculate top1 acc index:{}, zj_embedding shape:{}'.format(idx, zj_embedding.shape))
    args.logger.info('INFO, calculate top1 acc index:{}, jk_all_embedding shape:{}'.format(idx, jk_all_embedding.shape))

    test_time = time.time()
    mm = np.matmul(np.expand_dims(zj_embedding, axis=0), dis_embedding_tot)
    top100_jk2zj = np.squeeze(topk(mm, 100)[0], axis=0)
    top100_zj2jk = topk(np.matmul(jk_all_embedding, dis_embedding_tot), 100)[0]
    test_time_used = time.time() - test_time
    args.logger.info('INFO, calculate top1 acc index:{}, np.matmul().top(100) time used:{:.2f}s'.format(
        idx, test_time_used))
    tot[0] = len(jk_all)
    for i, jk in enumerate(jk_all):
        jk_embedding = test_embedding_tot[jk]
        similarity = np.dot(jk_embedding, zj_embedding)
        if similarity > top100_jk2zj[0]:
            correct[0] += 1
        if similarity > top100_zj2jk[i, 0]:
            correct[1] += 1
    return correct, tot

def l2normalize(features):
    epsilon = 1e-12
    l2norm = np.sum(np.abs(features) ** 2, axis=1, keepdims=True) ** (1./2)
    l2norm[np.logical_and(l2norm < 0, l2norm > -epsilon)] = -epsilon
    l2norm[np.logical_and(l2norm >= 0, l2norm < epsilon)] = epsilon
    return features/l2norm

def main(args):
    if not os.path.exists(args.test_dir):
        args.logger.info('ERROR, test_dir is not exists, please set test_dir in config.py.')
        return 0
    all_start_time = time.time()

    net = get_model(args)
    compile_time_used = time.time() - all_start_time
    args.logger.info('INFO, graph compile finished, time used:{:.2f}s, start calculate img embedding'.
                     format(compile_time_used))

    img_transforms = transforms.Compose([vision.ToTensor(), vision.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



    #for test images
    args.logger.info('INFO, start step1, calculate test img embedding, weight file = {}'.format(args.weight))
    step1_start_time = time.time()

    ds, img_tot, all_labels = get_dataloader(args.test_img_predix, args.test_img_list,
                                             args.test_batch_size, img_transforms)
    args.logger.info('INFO, dataset total test img:{}, total test batch:{}'.format(img_tot, ds.get_dataset_size()))
    test_embedding_tot_np = np.zeros((img_tot, args.emb_size))
    test_img_labels = all_labels
    data_loader = ds.create_dict_iterator(output_numpy=True, num_epochs=1)
    for i, data in enumerate(data_loader):
        img, idxs = data["image"], data["index"]
        out = net(Tensor(img)).asnumpy().astype(np.float32)
        embeddings = l2normalize(out)
        for batch in range(embeddings.shape[0]):
            test_embedding_tot_np[idxs[batch]] = embeddings[batch]
    try:
        check_minmax(args, np.linalg.norm(test_embedding_tot_np, ord=2, axis=1))
    except ValueError:
        return 0

    test_embedding_tot = {}
    for idx, label in enumerate(test_img_labels):
        test_embedding_tot[label] = test_embedding_tot_np[idx]

    step2_start_time = time.time()
    step1_time_used = step2_start_time - step1_start_time
    args.logger.info('INFO, step1 finished, time used:{:.2f}s, start step2, calculate dis img embedding'.
                     format(step1_time_used))

    # for dis images
    ds_dis, img_tot, _ = get_dataloader(args.dis_img_predix, args.dis_img_list, args.dis_batch_size, img_transforms)
    dis_embedding_tot_np = np.zeros((img_tot, args.emb_size))
    total_batch = ds_dis.get_dataset_size()
    args.logger.info('INFO, dataloader total dis img:{}, total dis batch:{}'.format(img_tot, total_batch))
    start_time = time.time()
    img_per_gpu = int(math.ceil(1.0 * img_tot / args.world_size))
    delta_num = img_per_gpu * args.world_size - img_tot
    start_idx = img_per_gpu * args.local_rank - max(0, args.local_rank - (args.world_size - delta_num))
    data_loader = ds_dis.create_dict_iterator(output_numpy=True, num_epochs=1)
    for idx, data in enumerate(data_loader):
        img = data["image"]
        out = net(Tensor(img)).asnumpy().astype(np.float32)
        embeddings = l2normalize(out)
        dis_embedding_tot_np[start_idx:(start_idx + embeddings.shape[0])] = embeddings
        start_idx += embeddings.shape[0]
        if args.local_rank % 8 == 0 and idx % args.log_interval == 0 and idx > 0:
            speed = 1.0 * (args.dis_batch_size * args.log_interval * args.world_size) / (time.time() - start_time)
            time_left = (total_batch - idx - 1) * args.dis_batch_size *args.world_size / speed
            args.logger.info('INFO, processed [{}/{}], speed: {:.2f} img/s, left:{:.2f}s'.
                             format(idx, total_batch, speed, time_left))
            start_time = time.time()
    try:
        check_minmax(args, np.linalg.norm(dis_embedding_tot_np, ord=2, axis=1))
    except ValueError:
        return 0

    step3_start_time = time.time()
    step2_time_used = step3_start_time - step2_start_time
    args.logger.info('INFO, step2 finished, time used:{:.2f}s, start step3, calculate top1 acc'.format(step2_time_used))

    # clear npu memory

    img = None
    net = None

    dis_embedding_tot_np = np.transpose(dis_embedding_tot_np, (1, 0))
    args.logger.info('INFO, calculate top1 acc dis_embedding_tot_np shape:{}'.format(dis_embedding_tot_np.shape))

    # find best match
    assert len(args.test_img_list) % 2 == 0
    task_num = int(len(args.test_img_list) / 2)
    correct = np.array([0] * (2 * task_num))
    tot = np.array([0] * task_num)

    for i in range(int(len(args.test_img_list) / 2)):
        jk_list = args.test_img_list[2 * i]
        zj_list = args.test_img_list[2 * i + 1]
        zj2jk_pairs = sorted(generate_test_pair(jk_list, zj_list))
        sampler = DistributedSampler(zj2jk_pairs)
        args.logger.info('INFO, calculate top1 acc sampler len:{}'.format(len(sampler)))
        for idx in sampler:
            out1, out2 = cal_topk(args, idx, zj2jk_pairs, test_embedding_tot, dis_embedding_tot_np)
            correct[2 * i] += out1[0]
            correct[2 * i + 1] += out1[1]
            tot[i] += out2[0]

    args.logger.info('local_rank={},tot={},correct={}'.format(args.local_rank, tot, correct))

    step3_time_used = time.time() - step3_start_time
    args.logger.info('INFO, step3 finished, time used:{:.2f}s'.format(step3_time_used))
    args.logger.info('weight:{}'.format(args.weight))

    for i in range(int(len(args.test_img_list) / 2)):
        test_set_name = 'test_dataset'
        zj2jk_acc = correct[2 * i] / tot[i]
        jk2zj_acc = correct[2 * i + 1] / tot[i]
        avg_acc = (zj2jk_acc + jk2zj_acc) / 2
        results = '[{}]: zj2jk={:.4f}, jk2zj={:.4f}, avg={:.4f}'.format(test_set_name, zj2jk_acc, jk2zj_acc, avg_acc)
        args.logger.info(results)
    args.logger.info('INFO, tot time used: {:.2f}s'.format(time.time() - all_start_time))
    return 0

if __name__ == '__main__':
    arg = config_inference
    arg.test_img_predix = [arg.test_dir, arg.test_dir]

    arg.test_img_list = [os.path.join(arg.test_dir, 'lists/jk_list.txt'),
                         os.path.join(arg.test_dir, 'lists/zj_list.txt')]
    arg.dis_img_predix = [arg.test_dir,]
    arg.dis_img_list = [os.path.join(arg.test_dir, 'lists/dis_list.txt'),]

    log_path = os.path.join(arg.ckpt_path, 'logs')
    arg.logger = get_logger(log_path, arg.local_rank)

    arg.logger.info('Config\n\n{}\n'.format(pformat(arg)))

    main(arg)
