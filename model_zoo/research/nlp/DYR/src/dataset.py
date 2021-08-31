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
"""
Data operations, will be used in run_pretrain.py
"""
import os
import random
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C

# samples in one block
POS_SIZE = 1
# max seq length
SEQ_LEN = 512
# rand id
RANK_ID = 0
# pos and neg samples in one minibatch on one device id
GROUP_SIZE = 8
# group number
GROUP_NUM = 1
# device number
DEVICE_NUM = 1
# batch size
BATCH_SIZE = 1

def process_samples_base(input_ids, input_mask, segment_ids, label_ids):
    """create block of samples"""
    random.seed(1)
    global GROUP_SIZE, SEQ_LEN
    neg_len = GROUP_SIZE - 1
    input_ids = input_ids.reshape(-1, SEQ_LEN)
    input_mask = input_mask.reshape(-1, SEQ_LEN)
    segment_ids = segment_ids.reshape(-1, SEQ_LEN)
    label_ids = label_ids.reshape(-1, 1)
    input_ids_l = input_ids.tolist()
    input_mask_l = input_mask.tolist()
    segment_ids_l = segment_ids.tolist()
    label_ids_l = label_ids.tolist()

    temp = []
    for i in range(1, len(input_ids_l)):
        temp.append({"input_ids": input_ids_l[i],
                     "input_mask": input_mask_l[i],
                     "segment_ids": segment_ids_l[i],
                     "label_ids": label_ids_l[i]})
    negs = []
    if len(temp) < neg_len:
        negs = random.choices(temp, k=neg_len)
    else:
        negs = random.sample(temp, k=neg_len)
    input_ids_n = [input_ids_l.pop(0)]
    input_mask_n = [input_mask_l.pop(0)]
    segment_ids_n = [segment_ids_l.pop(0)]
    label_ids_n = [label_ids_l.pop(0)]
    for i in range(neg_len):
        input_ids_n.append(negs[i]["input_ids"])
        input_mask_n.append(negs[i]["input_mask"])
        segment_ids_n.append(negs[i]["segment_ids"])
        label_ids_n.append(negs[i]["label_ids"])
    input_ids = np.array(input_ids_n, dtype=np.int64)
    input_mask = np.array(input_mask_n, dtype=np.int64)
    segment_ids = np.array(segment_ids_n, dtype=np.int64)
    label_ids = np.array(label_ids_n, dtype=np.int64)

    input_ids = input_ids.reshape(-1, SEQ_LEN)
    input_mask = input_mask.reshape(-1, SEQ_LEN)
    segment_ids = segment_ids.reshape(-1, SEQ_LEN)
    label_ids = label_ids.reshape(-1, POS_SIZE)
    return input_ids, input_mask, segment_ids, label_ids

def samples_base(input_ids, input_mask, segment_ids, label_ids):
    """split samples for device"""
    global GROUP_SIZE, GROUP_NUM, RANK_ID, SEQ_LEN, BATCH_SIZE, DEVICE_NUM
    out_ids = []
    out_mask = []
    out_seg = []
    out_label = []
    assert len(input_ids) >= len(input_mask)
    assert len(input_ids) >= len(segment_ids)
    assert len(input_ids) >= len(label_ids)
    group_id = RANK_ID * GROUP_NUM // DEVICE_NUM
    begin0 = BATCH_SIZE * group_id
    end0 = (group_id + 1) * BATCH_SIZE
    begin = (RANK_ID % (DEVICE_NUM//GROUP_NUM)) * GROUP_NUM * GROUP_SIZE // DEVICE_NUM
    end = ((RANK_ID % (DEVICE_NUM//GROUP_NUM)) + 1) * GROUP_NUM * GROUP_SIZE // DEVICE_NUM
    begin_temp = begin
    end_temp = end
    for i in range(begin0, end0):
        ids, mask, seg, lab = input_ids[i], input_mask[i], segment_ids[i], label_ids[i]
        if begin_temp > len(input_ids[i]):
            begin_temp = begin_temp - len(input_ids[i])
            end_temp = end_temp - len(input_ids[i])
            continue
        ids = ids.reshape(-1, SEQ_LEN)
        mask = mask.reshape(-1, SEQ_LEN)
        seg = seg.reshape(-1, SEQ_LEN)
        lab = lab.reshape(-1, 1)
        ids = ids[begin_temp:end_temp]
        mask = mask[begin_temp:end_temp]
        seg = seg[begin_temp:end_temp]
        lab = lab[begin_temp:end_temp]
        out_ids.append(ids)
        out_mask.append(mask)
        out_seg.append(seg)
        out_label.append(lab)
        begin_temp = begin
        end_temp = end
    input_ids = np.array(out_ids, dtype=np.int64)
    input_mask = np.array(out_mask, dtype=np.int64)
    segment_ids = np.array(out_seg, dtype=np.int64)
    label_ids = np.array(out_label, dtype=np.int64)
    return input_ids, input_mask, segment_ids, label_ids

def create_dyr_base_dataset(device_num=1, rank=0, batch_size=1, repeat_count=1, dataset_format="mindrecord",
                            data_file_path=None, schema_file_path=None, do_shuffle=True,
                            group_size=1, group_num=1, seq_len=512):
    """create finetune dataset"""
    global GROUP_SIZE, GROUP_NUM, RANK_ID, SEQ_LEN, BATCH_SIZE, DEVICE_NUM
    GROUP_SIZE = group_size
    GROUP_NUM = group_num
    RANK_ID = rank
    SEQ_LEN = seq_len
    BATCH_SIZE = batch_size
    DEVICE_NUM = device_num
    print("device_num = %d, rank_id = %d, batch_size = %d" %(device_num, rank, batch_size))
    print("group_size = %d, group_num = %d, seq_len = %d" %(group_size, group_num, seq_len))

    divide = (group_size * group_num) % device_num
    assert divide == 0
    assert device_num >= group_num
    type_cast_op = C.TypeCast(mstype.int32)
    ds.config.set_seed(1000)
    random.seed(1)
    data_files = []
    if ".mindrecord" in data_file_path:
        data_files.append(data_file_path)
    else:
        files = os.listdir(data_file_path)
        for file_name in files:
            if "mindrecord" in file_name and "mindrecord.db" not in file_name:
                data_files.append(os.path.join(data_file_path, file_name))

    if dataset_format == "mindrecord":
        data_set = ds.MindDataset(data_files,
                                  columns_list=["input_ids", "input_mask", "segment_ids", "label_ids"],
                                  shuffle=do_shuffle)
    else:
        data_set = ds.TFRecordDataset([data_file_path], schema_file_path if schema_file_path != "" else None,
                                      columns_list=["input_ids", "input_mask", "segment_ids", "label_ids"],
                                      shuffle=do_shuffle)

    data_set = data_set.map(operations=process_samples_base,
                            input_columns=["input_ids", "input_mask", "segment_ids", "label_ids"])
    batch_size = batch_size * group_num
    data_set = data_set.batch(batch_size, drop_remainder=True)
    data_set = data_set.map(operations=samples_base,
                            input_columns=["input_ids", "input_mask", "segment_ids", "label_ids"])
    data_set = data_set.map(operations=type_cast_op, input_columns="label_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="segment_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_mask")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_ids")
    data_set = data_set.repeat(repeat_count)
    # apply batch operations
    return data_set

def process_samples(input_ids, input_mask, segment_ids, label_ids):
    """create block of samples"""
    random.seed(None)
    rand_id = random.sample(range(0, 15), 15)
    random.seed(1)
    global GROUP_SIZE, SEQ_LEN
    neg_len = GROUP_SIZE - 1
    input_ids = input_ids.reshape(-1, SEQ_LEN)
    input_mask = input_mask.reshape(-1, SEQ_LEN)
    segment_ids = segment_ids.reshape(-1, SEQ_LEN)
    label_ids = label_ids.reshape(-1, 1)
    input_ids_l = input_ids.tolist()
    input_mask_l = input_mask.tolist()
    segment_ids_l = segment_ids.tolist()
    label_ids_l = label_ids.tolist()

    temp = []
    for i in range(1, len(input_ids_l)):
        temp.append({"input_ids": input_ids_l[i],
                     "input_mask": input_mask_l[i],
                     "segment_ids": segment_ids_l[i],
                     "label_ids": label_ids_l[i]})
    negs = []
    if len(temp) < neg_len:
        negs = random.choices(temp, k=neg_len)
    else:
        negs = random.sample(temp, k=neg_len)
    input_ids_n = [input_ids_l.pop(0)]
    input_mask_n = [input_mask_l.pop(0)]
    segment_ids_n = [segment_ids_l.pop(0)]
    label_ids_n = [label_ids_l.pop(0)]
    for i in range(neg_len):
        input_ids_n.append(negs[i]["input_ids"])
        input_mask_n.append(negs[i]["input_mask"])
        segment_ids_n.append(negs[i]["segment_ids"])
        label_ids_n.append(negs[i]["label_ids"])
    input_ids = np.array(input_ids_n, dtype=np.int64)
    input_mask = np.array(input_mask_n, dtype=np.int64)
    segment_ids = np.array(segment_ids_n, dtype=np.int64)
    label_ids = np.array(label_ids_n, dtype=np.int64)

    input_ids = input_ids.reshape(-1, SEQ_LEN)
    input_mask = input_mask.reshape(-1, SEQ_LEN)
    segment_ids = segment_ids.reshape(-1, SEQ_LEN)
    label_ids = label_ids.reshape(-1, POS_SIZE)

    label_ids = np.array(rand_id, dtype=np.int64)
    label_ids = label_ids.reshape(-1, 15)
    return input_ids, input_mask, segment_ids, label_ids

def samples(input_ids, input_mask, segment_ids, label_ids):
    """split samples for device"""
    global GROUP_SIZE, GROUP_NUM, RANK_ID, SEQ_LEN, BATCH_SIZE, DEVICE_NUM
    out_ids = []
    out_mask = []
    out_seg = []
    out_label = []
    assert len(input_ids) >= len(input_mask)
    assert len(input_ids) >= len(segment_ids)
    assert len(input_ids) >= len(label_ids)
    group_id = RANK_ID * GROUP_NUM // DEVICE_NUM
    begin0 = BATCH_SIZE * group_id
    end0 = (group_id + 1) * BATCH_SIZE
    begin = (RANK_ID % (DEVICE_NUM // GROUP_NUM)) * GROUP_NUM * GROUP_SIZE // DEVICE_NUM
    end = ((RANK_ID % (DEVICE_NUM // GROUP_NUM)) + 1) * GROUP_NUM * GROUP_SIZE // DEVICE_NUM
    begin_temp = begin
    end_temp = end
    for i in range(begin0, end0):
        ids, mask, seg, lab = input_ids[i], input_mask[i], segment_ids[i], label_ids[i]
        if begin_temp > len(input_ids[i]):
            begin_temp = begin_temp - len(input_ids[i])
            end_temp = end_temp - len(input_ids[i])
            continue
        ids = ids.reshape(-1, SEQ_LEN)
        mask = mask.reshape(-1, SEQ_LEN)
        seg = seg.reshape(-1, SEQ_LEN)
        lab = lab.reshape(-1, 15)
        ids = ids[begin_temp:end_temp]
        mask = mask[begin_temp:end_temp]
        seg = seg[begin_temp:end_temp]
        out_ids.append(ids)
        out_mask.append(mask)
        out_seg.append(seg)
        out_label.append(lab)
        begin_temp = begin
        end_temp = end
    input_ids = np.array(out_ids, dtype=np.int64)
    input_mask = np.array(out_mask, dtype=np.int64)
    segment_ids = np.array(out_seg, dtype=np.int64)
    label_ids = np.array(out_label, dtype=np.int64)
    return input_ids, input_mask, segment_ids, label_ids

def create_dyr_dataset(device_num=1, rank=0, batch_size=1, repeat_count=1, dataset_format="mindrecord",
                       data_file_path=None, schema_file_path=None, do_shuffle=True,
                       group_size=1, group_num=1, seq_len=512):
    """create finetune dataset"""
    global GROUP_SIZE, GROUP_NUM, RANK_ID, SEQ_LEN, BATCH_SIZE, DEVICE_NUM
    GROUP_SIZE = group_size
    GROUP_NUM = group_num
    RANK_ID = rank
    SEQ_LEN = seq_len
    BATCH_SIZE = batch_size
    DEVICE_NUM = device_num
    print("device_num = %d, rank_id = %d, batch_size = %d" %(device_num, rank, batch_size))
    print("group_size = %d, group_num = %d, seq_len = %d" %(group_size, group_num, seq_len))

    divide = (group_size * group_num) % device_num
    assert divide == 0
    assert device_num >= group_num
    type_cast_op = C.TypeCast(mstype.int32)
    ds.config.set_seed(1000)
    random.seed(1)
    data_files = []
    if ".mindrecord" in data_file_path:
        data_files.append(data_file_path)
    else:
        files = os.listdir(data_file_path)
        for file_name in files:
            if "mindrecord" in file_name and "mindrecord.db" not in file_name:
                data_files.append(os.path.join(data_file_path, file_name))

    if dataset_format == "mindrecord":
        data_set = ds.MindDataset(data_files,
                                  columns_list=["input_ids", "input_mask", "segment_ids", "label_ids"],
                                  shuffle=do_shuffle)
    else:
        data_set = ds.TFRecordDataset([data_file_path], schema_file_path if schema_file_path != "" else None,
                                      columns_list=["input_ids", "input_mask", "segment_ids", "label_ids"],
                                      shuffle=do_shuffle)

    data_set = data_set.map(operations=process_samples,
                            input_columns=["input_ids", "input_mask", "segment_ids", "label_ids"])
    batch_size = batch_size * group_num
    data_set = data_set.batch(batch_size, drop_remainder=True)
    data_set = data_set.map(operations=samples, input_columns=["input_ids", "input_mask", "segment_ids", "label_ids"])
    data_set = data_set.map(operations=type_cast_op, input_columns="label_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="segment_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_mask")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_ids")
    data_set = data_set.repeat(repeat_count)
    return data_set
def create_dyr_dataset_predict(batch_size=1, repeat_count=1, dataset_format="mindrecord",
                               data_file_path=None, schema_file_path=None, do_shuffle=True):
    """create evaluation dataset"""
    type_cast_op = C.TypeCast(mstype.int32)
    data_set = ds.MindDataset([data_file_path],
                              columns_list=["input_ids", "input_mask", "segment_ids"],
                              shuffle=do_shuffle)
    data_set = data_set.map(operations=type_cast_op, input_columns="segment_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_mask")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_ids")
    data_set = data_set.repeat(repeat_count)
    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)
    return data_set
