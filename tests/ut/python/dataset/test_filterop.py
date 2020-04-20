# Copyright 2019 Huawei Technologies Co., Ltd
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
# ==============================================================================

import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.transforms.vision.c_transforms as cde
import mindspore.dataset.transforms.c_transforms as C
import mindspore.common.dtype as mstype
from mindspore import log as logger

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
# test for predicate
def test_diff_predicate_func():
    def test_filter(predicate_func):
        transforms = [
            cde.Decode(),
            cde.Resize([64, 64])
        ]
        type_cast_op = C.TypeCast(mstype.int32)
        dataset = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image", "label"], shuffle=False)
        dataset = dataset.map(input_columns=["image"], operations=transforms, num_parallel_workers=1)
        dataset = dataset.filter(input_columns=["image", "label"], predicate=predicate_func, num_parallel_workers=4)

        num_iter = 0
        label_list = []
        for data in dataset.create_dict_iterator():
            num_iter += 1
            ori_img = data["image"]
            label = data["label"]
            label_list.append(label)
        assert num_iter == 1
        assert label_list[0] == 3

    test_filter(lambda image, label: label == 3)
    test_filter(lambda image, label: label[0] == 3)
    test_filter(lambda image, label: label == [3])
    test_filter(lambda image, label: label == np.array([3]))
    test_filter(lambda image, label: label == np.array(3))

def filter_func_ge(data):
    if data > 10:
        return False
    return True


def generator_1d():
    for i in range(64):
        yield (np.array(i),)

# test with GeneratorDataset
def test_filter_by_generator_with_no():
    dataset = ds.GeneratorDataset(generator_1d, ["data"])
    dataset_f = dataset.filter(predicate=lambda data: data < 11, num_parallel_workers=4)
    num_iter = 0
    expected_rs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for item in dataset_f.create_dict_iterator():
        assert item["data"] == expected_rs[num_iter]
        num_iter += 1

# test with repeatOp before
def test_filter_by_generator_with_repeat():
    dataset = ds.GeneratorDataset(generator_1d, ["data"])
    dataset_r = dataset.repeat(4)
    dataset_f = dataset_r.filter(predicate=filter_func_ge, num_parallel_workers=4)
    num_iter = 0
    ret_data = []
    expected_rs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for item in dataset_f.create_dict_iterator():
        num_iter += 1
        ret_data.append(item["data"])
    assert num_iter == 44
    for i in range(4):
        for ii in range(len(expected_rs)):
            index =  i * len(expected_rs) + ii 
            assert  ret_data[index] == expected_rs[ii]

# test with repeatOp after
def test_filter_by_generator_with_repeat_after():
    dataset = ds.GeneratorDataset(generator_1d, ["data"])
    dataset_f = dataset.filter(predicate=filter_func_ge, num_parallel_workers=4)
    dataset_r = dataset_f.repeat(4)
    num_iter = 0
    ret_data = []
    expected_rs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for item in dataset_r.create_dict_iterator():
        num_iter += 1
        ret_data.append(item["data"])
    assert num_iter == 44
    for i in range(4):
        for ii in range(len(expected_rs)):
            index =  i * len(expected_rs) + ii 
            assert  ret_data[index] == expected_rs[ii]

def filter_func_batch(data):
    if data[0] > 8:
        return False
    return True

def filter_func_batch_after(data):
    if data > 20:
        return False
    return True

# test with batchOp before
def test_filter_by_generator_with_batch():
    dataset = ds.GeneratorDataset(generator_1d, ["data"])
    dataset_b = dataset.batch(4)
    dataset_f = dataset_b.filter(predicate=filter_func_batch, num_parallel_workers=4)
    num_iter = 0
    ret_data = []
    for item in dataset_f.create_dict_iterator():
        num_iter += 1
        ret_data.append(item["data"])
    assert num_iter == 3
    assert ret_data[0][0] == 0
    assert ret_data[1][0] == 4
    assert ret_data[2][0] == 8

# test with batchOp after
def test_filter_by_generator_with_batch_after():
    dataset = ds.GeneratorDataset(generator_1d, ["data"])
    dataset_f = dataset.filter(predicate=filter_func_batch_after, num_parallel_workers=4)
    dataset_b = dataset_f.batch(4)
    num_iter = 0
    ret_data = []
    for item in dataset_b.create_dict_iterator():
        num_iter += 1
        ret_data.append(item["data"])
    assert num_iter == 6
    assert ret_data[0][0] == 0
    assert ret_data[1][0] == 4
    assert ret_data[5][0] == 20


def filter_func_shuffle(data):
    if data > 20:
        return False
    return True

# test with batchOp before
def test_filter_by_generator_with_shuffle():
    dataset = ds.GeneratorDataset(generator_1d, ["data"])
    dataset_s = dataset.shuffle(4)
    dataset_f = dataset_s.filter(predicate=filter_func_shuffle, num_parallel_workers=4)
    num_iter = 0
    for item in dataset_f.create_dict_iterator():
            num_iter += 1
    assert num_iter == 21


def filter_func_shuffle_after(data):
    if data > 20:
        return False
    return True

# test with batchOp after
def test_filter_by_generator_with_shuffle_after():
    dataset = ds.GeneratorDataset(generator_1d, ["data"])
    dataset_f = dataset.filter(predicate=filter_func_shuffle_after, num_parallel_workers=4)
    dataset_s = dataset_f.shuffle(4)
    num_iter = 0
    for item in dataset_s.create_dict_iterator():
        num_iter += 1
    assert num_iter == 21


def generator_1d_zip1():
    for i in range(64):
        yield (np.array(i),)


def generator_1d_zip2():
    for i in range(64):
        yield (np.array(i+100),)


def filter_func_zip(data1, data2):
    if data1 >  20:
        return False
    return True

def filter_func_zip_after(data1):
    if data1 >  20:
        return False
    return True

# test with zipOp before
def test_filter_by_generator_with_zip():
    dataset1 = ds.GeneratorDataset(generator_1d_zip1, ["data1"])
    dataset2 = ds.GeneratorDataset(generator_1d_zip2, ["data2"])
    dataz = ds.zip((dataset1, dataset2))
    dataset_f = dataz.filter(predicate=filter_func_zip, num_parallel_workers=1)
    num_iter = 0
    ret_data = []
    for item in dataset_f.create_dict_iterator():
        num_iter += 1
        ret_data.append({"data1": item["data1"], "data2":item["data2"]})
    assert num_iter == 21
    assert ret_data[0]["data1"] ==  0
    assert ret_data[0]["data2"] ==  100
    assert ret_data[5]["data1"] ==  5
    assert ret_data[5]["data2"] ==  105


# test with zipOp after
def test_filter_by_generator_with_zip_after():
    dataset1 = ds.GeneratorDataset(generator_1d_zip1, ["data1"])
    dataset2 = ds.GeneratorDataset(generator_1d_zip1, ["data2"])
    dt1 = dataset1.filter(predicate=filter_func_zip_after, num_parallel_workers=4)
    dt2 = dataset2.filter(predicate=filter_func_zip_after, num_parallel_workers=4)
    dataz = ds.zip((dt1, dt2))
    num_iter = 0
    ret_data = []
    for item in dataz.create_dict_iterator():
        num_iter += 1
        ret_data.append({"data1": item["data1"], "data2":item["data2"]})
    assert num_iter == 21
    assert ret_data[0]["data1"] ==  0
    assert ret_data[0]["data2"] ==  0
    assert ret_data[5]["data1"] ==  5
    assert ret_data[5]["data2"] ==  5


def filter_func_map(col1, col2):
    if col1[0] > 8:
        return True
    return False


def filter_func_map_part(col1):
    if col1 < 3:
       return True
    else:
        return False


def filter_func_map_all(col1, col2):
    return True

def generator_mc(maxid=20):
    for i in range(maxid):
        yield (np.array([i]), np.array([[i, i + 1], [i + 2, i + 3]]))


def func_map(data_col1, data_col2):
    return (data_col1, data_col2)


def func_map_part(data_col1):
    return (data_col1)

# test with  map
def test_filter_by_generator_with_map_all_col():
    dataset = ds.GeneratorDataset(generator_mc(12), ["col1", "col2"])
    dataset_map = dataset.map( input_columns=["col1"],  output_columns=["col1"] , operations=func_map_part)
    # dataset_map = dataset.map(  operations=func_map_part)
    dataset_f = dataset_map.filter(input_columns=["col1"],  predicate=filter_func_map_part, num_parallel_workers=1)
    num_iter = 0
    ret_data = []
    for item in dataset_f.create_dict_iterator():
        num_iter += 1
        ret_data.append(item["col1"])
    assert num_iter == 3
    assert ret_data[0] == 0
    assert ret_data[1] == 1

# test with  map
def test_filter_by_generator_with_map_part_col():
    dataset = ds.GeneratorDataset(generator_mc(12), ["col1", "col2"])
    dataset_map = dataset.map( input_columns=["col1"],  output_columns=["out1"] , operations=func_map_part)
    
    dataset_f = dataset_map.filter(input_columns=["out1", "col2"], predicate=filter_func_map, num_parallel_workers=4)
    num_iter = 0
    ret_data = []
    for item in dataset_f.create_dict_iterator():
            num_iter += 1
            print(item)
            ret_data.append(item["out1"])
    assert num_iter == 3
    assert ret_data[0] == 9
    assert ret_data[2] == 11


def filter_func_rename(data):
    if data>  8:
        return True
    return False

# test with  rename before
def test_filter_by_generator_with_rename():
    dataset = ds.GeneratorDataset(generator_1d, ["data"])
    dataset_b = dataset.rename(input_columns=["data"], output_columns=["col1"])
    dataset_f = dataset_b.filter(predicate=filter_func_rename, num_parallel_workers=4)
    num_iter = 0
    ret_data = []
    for item in dataset_f.create_dict_iterator():
        num_iter += 1
        ret_data.append(item["col1"])
    assert num_iter == 55
    assert ret_data[0] == 9
    assert ret_data[54] == 63


#test input_column
def filter_func_input_column1(col1, col2):
    if col1[0] < 8:
        return True
    return False

def filter_func_input_column2(col1):
    if col1[0] < 8:
        return True
    return False

def filter_func_input_column3(col1):
    return True

# test with  input_columns
def test_filter_by_generator_with_input_column():
    dataset = ds.GeneratorDataset(generator_mc(64), ["col1", "col2"])
    dataset_map = dataset.map( input_columns=["col1"],  output_columns=["out1"] , operations=func_map_part)
    dataset_f1 = dataset_map.filter(input_columns=["out1", "col2"], predicate=filter_func_input_column1, num_parallel_workers=4)
    dataset_f2 = dataset_f1.filter(input_columns=["out1"], predicate=filter_func_input_column2, num_parallel_workers=4)
    dataset_f3 = dataset_f2.filter(input_columns=["col2"],  predicate=filter_func_input_column3, num_parallel_workers=4)
    dataset_f4 = dataset_f3.filter(predicate=filter_func_input_column1, num_parallel_workers=4)
    num_iter = 0
    ret_data = []
    for item in dataset_f4.create_dict_iterator():
            num_iter += 1
            ret_data.append(item["out1"])
    assert num_iter == 8
    assert ret_data[0] == 0
    assert ret_data[7] == 7


#test kFilterPartial
def generator_mc_p0(maxid=20):
    for i in range(maxid):
        yield (np.array([i ]), np.array([i + 100]))

def generator_mc_p1(maxid=20):
    for i in range(maxid):
        yield (np.array([i + 200 ]), np.array([i + 300]))


def filter_func_Partial_0(col1, col2, col3, col4):
    filter_data = [0,1,2,3,4,  11]
    if col1[0]  in filter_data:
        return False
    return True

# test with  row_data_buffer > 1
def test_filter_by_generator_Partial0():
    ds.config.load('../data/dataset/declient_filter.cfg')
    dataset1= ds.GeneratorDataset(source = generator_mc_p0(), column_names = ["col1", "col2"])
    dataset2 = ds.GeneratorDataset(source = generator_mc_p1(), column_names = ["col3", "col4"])
    dataset_zip = ds.zip((dataset1, dataset2))
    dataset_f1 = dataset_zip.filter(predicate=filter_func_Partial_0, num_parallel_workers=2)
    ret = []
    for item in dataset_f1.create_dict_iterator():
           ret.append(item["col1"])
    assert ret[0] == 5
    assert ret[6] == 12

# test with  row_data_buffer > 1
def test_filter_by_generator_Partial1():
    ds.config.load('../data/dataset/declient_filter.cfg')
    dataset1= ds.GeneratorDataset(source = generator_mc_p0(), column_names = ["col1", "col2"])
    dataset2 = ds.GeneratorDataset(source = generator_mc_p1(), column_names = ["col3", "col4"])    
    dataset_zip = ds.zip((dataset1, dataset2))
    dataset_f1 = dataset_zip.filter(predicate=filter_func_Partial_0, num_parallel_workers=2)
    dataset_map = dataset_f1.map( input_columns=["col1"],  output_columns=["out1"] , operations=lambda x1: x1 + 400)
    ret = []
    for item in dataset_map.create_dict_iterator():
        ret.append(item["out1"])
    assert ret[0] == 405
    assert ret[6] == 412

# test with  row_data_buffer > 1
def test_filter_by_generator_Partial2():
    ds.config.load('../data/dataset/declient_filter.cfg')
    dataset1= ds.GeneratorDataset(source = generator_mc_p0(), column_names = ["col1", "col2"])
    dataset2 = ds.GeneratorDataset(source = generator_mc_p1(), column_names = ["col3", "col4"]) 

    dataset1f = dataset1.filter( input_columns= ["col1"],  predicate=lambda x:  x not in [3,7,9],  num_parallel_workers=2)
    dataset2f = dataset2.filter( input_columns= ["col3"],  predicate=lambda x:  x not in [203,207,209],  num_parallel_workers=2)
    dataset_zip = ds.zip((dataset1f, dataset2f))
    dataset_map = dataset_zip.map( input_columns=["col1", "col3"],  output_columns=["out1", "out3"] , operations=lambda x1,x3: (x1 + 400, x3+500))
    ret1 = []
    ret3 = []
    for item in dataset_map.create_dict_iterator():
        ret1.append(item["out1"])
        ret3.append(item["out3"])
    assert ret1[0] == 400
    assert ret1[6] == 408
    assert ret3[0] == 700
    assert ret3[6] == 708


def filter_func_Partial(col1, col2):
    if col1[0]  % 3 == 0:
        return True
    return False

def generator_big(maxid=20):
    for i in range(maxid):
        yield (np.array([i]), np.array([[i, i + 1], [i + 2, i + 3]]))

# test with  row_data_buffer > 1
def test_filter_by_generator_Partial():
    ds.config.load('../data/dataset/declient_filter.cfg')
    dataset = ds.GeneratorDataset(source = generator_mc(99), column_names = ["col1", "col2"])
    dataset_s = dataset.shuffle(4)
    dataset_f1 = dataset_s.filter(input_columns=["col1", "col2"], predicate=filter_func_Partial, num_parallel_workers=1)

    for item in dataset_f1.create_dict_iterator():
           assert item["col1"] % 3 == 0


def filter_func_cifar(col1, col2):
    if col2  % 3 == 0:
        return True
    return False

# test with  cifar10
def test_filte_case_dataset_cifar10():
    DATA_DIR_10 = "../data/dataset/testCifar10Data"
    ds.config.load('../data/dataset/declient_filter.cfg')
    dataset_c = ds.Cifar10Dataset(dataset_dir = DATA_DIR_10, num_samples = 100000,  shuffle=False)
    dataset_f1 = dataset_c.filter(input_columns=["image", "label"], predicate=filter_func_cifar, num_parallel_workers=1)
    num_iter = 0
    for item in dataset_f1.create_dict_iterator():
        # in this example, each dictionary has keys "image" and "label"
        assert item["label"] % 3 == 0

# column id sort 

def generator_sort1(maxid=20):
    for i in range(maxid):
        yield (np.array([i]), np.array([i + 100]), np.array([i + 200]))

def generator_sort2(maxid=20):
    for i in range(maxid):
        yield (np.array([i + 300]), np.array([i + 400]), np.array([i + 500])) 


def filter_func_part_sort(col1, col2, col3, col4, col5, col6):
    return True

def filter_func_map_sort(col1, col2, col3):
    return (col1, col2, col3)

def test_filter_by_generator_with_map_all_sort():
    dataset1 = ds.GeneratorDataset(generator_sort1(10), ["col1", "col2", "col3"])
    dataset2 = ds.GeneratorDataset(generator_sort2(10), ["col4 ", "col5", "col6"])

    dataz = ds.zip((dataset1, dataset2))
    dataset_f = dataz.filter(predicate=filter_func_part_sort, num_parallel_workers=1)
    num_iter = 0
    ret_data = []
    for item in dataset_f.create_dict_iterator():
        num_iter += 1
        ret_data.append(item)

    assert num_iter == 10
    assert ret_data[0]["col1"] == 0
    assert ret_data[9]["col6"] == 509



if __name__ == '__main__':
    test_diff_predicate_func()
    test_filte_case_dataset_cifar10()
    test_filter_by_generator_Partial0()
    test_filter_by_generator_Partial1()
    test_filter_by_generator_Partial2()
    test_filter_by_generator_with_batch()
    test_filter_by_generator_with_batch_after()
    test_filter_by_generator_with_input_column()
    test_filter_by_generator_with_map_all_col()
    test_filter_by_generator_with_map_all_sort()
    test_filter_by_generator_with_map_part_col()
    test_filter_by_generator_with_no()
    test_filter_by_generator_with_rename()
    test_filter_by_generator_with_repeat()
    test_filter_by_generator_with_repeat_after()
    test_filter_by_generator_with_shuffle()
    test_filter_by_generator_with_shuffle_after()
    test_filter_by_generator_with_zip()
    test_filter_by_generator_with_zip_after()
    test_filter_by_generator_Partial()
