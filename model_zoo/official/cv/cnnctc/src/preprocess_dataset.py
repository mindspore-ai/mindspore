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
"""preprocess dataset"""

import random
import pickle
import numpy as np
import lmdb
from tqdm import tqdm

def combine_lmdbs(lmdb_paths, lmdb_save_path):
    max_len = int((26 + 1) // 2)
    character = '0123456789abcdefghijklmnopqrstuvwxyz'

    env_save = lmdb.open(
        lmdb_save_path,
        map_size=1099511627776)

    cnt = 0
    for lmdb_path in lmdb_paths:
        env = lmdb.open(lmdb_path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            nSamples = nSamples

            # Filtering
            for index in tqdm(range(nSamples)):
                index += 1  # lmdb starts with 1
                label_key = 'label-%09d'.encode() % index
                label = txn.get(label_key).decode('utf-8')

                if len(label) > max_len:
                    continue

                illegal_sample = False
                for char_item in label.lower():
                    if char_item not in character:
                        illegal_sample = True
                        break
                if illegal_sample:
                    continue

                img_key = 'image-%09d'.encode() % index
                imgbuf = txn.get(img_key)

                with env_save.begin(write=True) as txn_save:
                    cnt += 1

                    label_key_save = 'label-%09d'.encode() % cnt
                    label_save = label.encode()
                    image_key_save = 'image-%09d'.encode() % cnt
                    image_save = imgbuf

                    txn_save.put(label_key_save, label_save)
                    txn_save.put(image_key_save, image_save)

    nSamples = cnt
    with env_save.begin(write=True) as txn_save:
        txn_save.put('num-samples'.encode(), str(nSamples).encode())


def analyze_lmdb_label_length(lmdb_path, batch_size=192, num_of_combinations=1000):
    label_length_dict = {}

    env = lmdb.open(lmdb_path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        nSamples = int(txn.get('num-samples'.encode()))
        nSamples = nSamples

        for index in tqdm(range(nSamples)):
            index += 1  # lmdb starts with 1
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')

            label_length = len(label)
            if label_length in label_length_dict:
                label_length_dict[label_length] += 1
            else:
                label_length_dict[label_length] = 1

    sorted_label_length = sorted(label_length_dict.items(), key=lambda x: x[1], reverse=True)

    label_length_sum = 0
    label_num = 0
    lengths = []
    p = []
    for l, num in sorted_label_length:
        label_length_sum += l * num
        label_num += num
        p.append(num)
        lengths.append(l)
    for i, _ in enumerate(p):
        p[i] /= label_num

    average_overall_length = int(label_length_sum / label_num * batch_size)

    def get_combinations_of_fix_length(fix_length, items, p, batch_size):
        ret = []
        cur_sum = 0
        ret = np.random.choice(items, batch_size - 1, True, p)
        cur_sum = sum(ret)
        ret = list(ret)
        if fix_length - cur_sum in items:
            ret.append(fix_length - cur_sum)
        else:
            return None
        return ret

    result = []
    while len(result) < num_of_combinations:
        ret = get_combinations_of_fix_length(average_overall_length, lengths, p, batch_size)
        if ret is not None:
            result.append(ret)
    return result


def generate_fix_shape_index_list(lmdb_path, combinations, pkl_save_path, num_of_iters=70000):
    length_index_dict = {}

    env = lmdb.open(lmdb_path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        nSamples = int(txn.get('num-samples'.encode()))
        nSamples = nSamples

        for index in tqdm(range(nSamples)):
            index += 1  # lmdb starts with 1
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')

            label_length = len(label)
            if label_length in length_index_dict:
                length_index_dict[label_length].append(index)
            else:
                length_index_dict[label_length] = [index]

    ret = []
    for _ in range(num_of_iters):
        comb = random.choice(combinations)
        for l in comb:
            ret.append(random.choice(length_index_dict[l]))

    with open(pkl_save_path, 'wb') as f:
        pickle.dump(ret, f, -1)


if __name__ == '__main__':
    # step 1: combine the SynthText dataset and MJSynth dataset into a single lmdb file
    print('Begin to combine multiple lmdb datasets')
    combine_lmdbs(['/home/workspace/mindspore_dataset/CNNCTC_Data/1_ST/',
                   '/home/workspace/mindspore_dataset/CNNCTC_Data/MJ_train/'],
                  '/home/workspace/mindspore_dataset/CNNCTC_Data/ST_MJ')

    # step 2: generate the order of input data, guarantee that the input batch shape is fixed
    print('Begin to generate the index order of input data')
    combination = analyze_lmdb_label_length('/home/workspace/mindspore_dataset/CNNCTC_Data/ST_MJ')
    generate_fix_shape_index_list('/home/workspace/mindspore_dataset/CNNCTC_Data/ST_MJ', combination,
                                  '/home/workspace/mindspore_dataset/CNNCTC_Data/st_mj_fixed_length_index_list.pkl')

    print('Done')
