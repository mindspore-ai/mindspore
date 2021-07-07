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

import os
import pickle
import numpy as np
from mindspore import dataset as ds
from mindspore.dataset.transforms import c_transforms as C
from mindspore.common.tensor import Tensor
from mindspore.common import dtype as mstype


class InputFeatures:
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, seq_length=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.seq_length = seq_length


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, cyclic_trunc=False):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: _ for _, label in enumerate(label_list)}

    features = []
    for example in examples:
        tokens = tokenizer.tokenize(example[0])
        seq_length = len(tokens)
        if seq_length > max_seq_length - 2:
            if cyclic_trunc:
                rand_index = np.random.randint(0, seq_length)
                tokens = [tokens[_] if _ < seq_length else tokens[_ - seq_length]
                          for _ in range(rand_index, rand_index + max_seq_length - 2)]
            else:
                tokens = tokens[: (max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        seq_length = len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        label_id = label_map[example[1]]

        features.append(InputFeatures(input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids,
                                      label_id=label_id,
                                      seq_length=seq_length))
    return features


def load_dataset(data_path, max_seq_length, tokenizer, batch_size, label_list=None, do_shuffle=True,
                 drop_remainder=True, output_dir=None, i=0, cyclic_trunc=False):
    if label_list is None:
        label_list = ['good', 'leimu', 'xiaoku', 'xin']
    with open(data_path, 'r', encoding='utf-8') as f:
        data = f.read()
    data_list = data.split('\n<<<')
    input_list = []
    for key in data_list[1:]:
        key = key.split('>>>')
        input_list.append([key[1], key[0]])
    datasets = create_ms_dataset(input_list, label_list, max_seq_length, tokenizer, batch_size,
                                 do_shuffle=do_shuffle, drop_remainder=drop_remainder, cyclic_trunc=cyclic_trunc)
    if output_dir is not None:
        output_path = os.path.join(output_dir, str(i) + '.dat')
        print(output_path)
        with open(output_path, "wb") as f:
            pickle.dump(tuple(datasets), f)
    del data, data_list, input_list
    return datasets, len(label_list)


def load_datasets(data_dir, max_seq_length, tokenizer, batch_size, label_list=None, do_shuffle=True,
                  drop_remainder=True, output_dir=None, cyclic_trunc=False):
    if label_list is None:
        label_list = ['good', 'leimu', 'xiaoku', 'xin']
    data_path_list = os.listdir(data_dir)
    datasets_list = []
    for i, relative_path in enumerate(data_path_list):
        data_path = os.path.join(data_dir, relative_path)
        with open(data_path, 'r', encoding='utf-8') as f:
            data = f.read()
        data_list = data.split('\n<<<')
        input_list = []
        for key in data_list[1:]:
            key = key.split('>>>')
            input_list.append([key[1], key[0]])
        datasets = create_ms_dataset(input_list, label_list, max_seq_length, tokenizer, batch_size,
                                     do_shuffle=do_shuffle, drop_remainder=drop_remainder, cyclic_trunc=cyclic_trunc)
        if output_dir is not None:
            output_path = os.path.join(output_dir, str(i) + '.dat')
            print(output_path)
            with open(output_path, "wb") as f:
                pickle.dump(tuple(datasets.create_tuple_iterator()), f)
        datasets_list.append(datasets)
    return datasets_list, len(label_list)


def create_ms_dataset(data_list, label_list, max_seq_length, tokenizer, batch_size, do_shuffle=True,
                      drop_remainder=True, cyclic_trunc=False):
    features = convert_examples_to_features(data_list, label_list, max_seq_length, tokenizer,
                                            cyclic_trunc=cyclic_trunc)

    def generator_func():
        for feature in features:
            yield (np.array(feature.input_ids),
                   np.array(feature.input_mask),
                   np.array(feature.segment_ids),
                   np.array(feature.label_id),
                   np.array(feature.seq_length))

    dataset = ds.GeneratorDataset(generator_func,
                                  ['input_ids', 'input_mask', 'token_type_id', 'label_ids', 'seq_length'])
    if do_shuffle:
        dataset = dataset.shuffle(buffer_size=10000)

    type_cast_op = C.TypeCast(mstype.int32)
    dataset = dataset.map(operations=[type_cast_op])
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return dataset


class ConstructMaskAndReplaceTensor:
    def __init__(self, batch_size, max_seq_length, vocab_size, keep_first_unchange=True, keep_last_unchange=True):
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        self.keep_first_unchange = keep_first_unchange
        self.keep_last_unchange = keep_last_unchange
        self.mask_tensor = np.ones((self.batch_size, self.max_seq_length))
        self.replace_tensor = np.zeros((self.batch_size, self.max_seq_length))

    def construct(self, seq_lengths):
        for i in range(self.batch_size):
            for j in range(seq_lengths[i]):
                rand1 = np.random.random()
                if rand1 < 0.15:
                    self.mask_tensor[i, j] = 0
                    rand2 = np.random.random()
                    if rand2 < 0.8:
                        self.replace_tensor[i, j] = 103
                    elif rand2 < 0.9:
                        self.mask_tensor[i, j] = 1
                    else:
                        self.replace_tensor[i, j] = np.random.randint(0, self.vocab_size)
                else:
                    self.mask_tensor[i, j] = 1
                    self.replace_tensor[i, j] = 0
            for j in range(seq_lengths[i], self.max_seq_length):
                self.mask_tensor[i, j] = 1
                self.replace_tensor[i, j] = 0
            if self.keep_first_unchange:
                self.mask_tensor[i, 0] = 1
                self.replace_tensor[i, 0] = 0
            if self.keep_last_unchange:
                self.mask_tensor[i, seq_lengths[i] - 1] = 1
                self.replace_tensor[i, seq_lengths[i] - 1] = 0
        mask_tensor = Tensor(self.mask_tensor, dtype=mstype.int32)
        replace_tensor = Tensor(self.replace_tensor, dtype=mstype.int32)
        return mask_tensor, replace_tensor
