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
# ==============================================================================
"""
Testing profiling support in DE
"""
import json
import os
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as vision

FILES = ["../data/dataset/testTFTestAllTypes/test.data"]
DATASET_ROOT = "../data/dataset/testTFTestAllTypes/"
SCHEMA_FILE = "../data/dataset/testTFTestAllTypes/datasetSchema.json"

PIPELINE_FILE = "./pipeline_profiling_1.json"
CPU_UTIL_FILE = "./minddata_cpu_utilization_1.json"
DATASET_ITERATOR_FILE = "./dataset_iterator_profiling_1.txt"


def set_profiling_env_var():
    """
    Set the MindData Profiling environment variables
    """
    os.environ['PROFILING_MODE'] = 'true'
    os.environ['MINDDATA_PROFILING_DIR'] = '.'
    os.environ['DEVICE_ID'] = '1'


def delete_profiling_files():
    """
    Delete the MindData profiling files generated from the test.
    Also disable the MindData Profiling environment variables.
    """
    # Delete MindData profiling files
    os.remove(PIPELINE_FILE)
    os.remove(CPU_UTIL_FILE)
    os.remove(DATASET_ITERATOR_FILE)

    # Disable MindData Profiling environment variables
    del os.environ['PROFILING_MODE']
    del os.environ['MINDDATA_PROFILING_DIR']
    del os.environ['DEVICE_ID']


def test_profiling_simple_pipeline():
    """
    Generator -> Shuffle -> Batch
    """
    set_profiling_env_var()

    source = [(np.array([x]),) for x in range(1024)]
    data1 = ds.GeneratorDataset(source, ["data"])
    data1 = data1.shuffle(64)
    data1 = data1.batch(32)
    # try output shape type and dataset size and make sure no profiling file is generated
    assert data1.output_shapes() == [[32, 1]]
    assert [str(tp) for tp in data1.output_types()] == ["int64"]
    assert data1.get_dataset_size() == 32

    # Confirm profiling files do not (yet) exist
    assert os.path.exists(PIPELINE_FILE) is False
    assert os.path.exists(CPU_UTIL_FILE) is False
    assert os.path.exists(DATASET_ITERATOR_FILE) is False

    try:
        for _ in data1:
            pass

        # Confirm profiling files now exist
        assert os.path.exists(PIPELINE_FILE) is True
        assert os.path.exists(CPU_UTIL_FILE) is True
        assert os.path.exists(DATASET_ITERATOR_FILE) is True

    except Exception as error:
        delete_profiling_files()
        raise error

    else:
        delete_profiling_files()


def test_profiling_complex_pipeline():
    """
    Generator -> Map     ->
                             -> Zip
    TFReader  -> Shuffle ->
    """
    set_profiling_env_var()

    source = [(np.array([x]),) for x in range(1024)]
    data1 = ds.GeneratorDataset(source, ["gen"])
    data1 = data1.map(operations=[(lambda x: x + 1)], input_columns=["gen"])

    pattern = DATASET_ROOT + "/test.data"
    data2 = ds.TFRecordDataset(pattern, SCHEMA_FILE, shuffle=ds.Shuffle.FILES)
    data2 = data2.shuffle(4)

    data3 = ds.zip((data1, data2))

    try:
        for _ in data3:
            pass

        with open(PIPELINE_FILE) as f:
            data = json.load(f)
            op_info = data["op_info"]
            assert len(op_info) == 5
            for i in range(5):
                if op_info[i]["op_type"] != "ZipOp":
                    assert "size" in op_info[i]["metrics"]["output_queue"]
                    assert "length" in op_info[i]["metrics"]["output_queue"]
                    assert "throughput" in op_info[i]["metrics"]["output_queue"]
                else:
                    # Note: Zip is an inline op and hence does not have metrics information
                    assert op_info[i]["metrics"] is None

    except Exception as error:
        delete_profiling_files()
        raise error

    else:
        delete_profiling_files()


def test_profiling_inline_ops_pipeline1():
    """
    Test pipeline with inline ops: Concat and EpochCtrl
    Generator ->
                 Concat -> EpochCtrl
    Generator ->
    """
    set_profiling_env_var()

    # In source1 dataset: Number of rows is 3; its values are 0, 1, 2
    def source1():
        for i in range(3):
            yield (np.array([i]),)

    # In source2 dataset: Number of rows is 7; its values are 3, 4, 5 ... 9
    def source2():
        for i in range(3, 10):
            yield (np.array([i]),)

    data1 = ds.GeneratorDataset(source1, ["col1"])
    data2 = ds.GeneratorDataset(source2, ["col1"])
    data3 = data1.concat(data2)

    try:
        # Note: If create_tuple_iterator() is called with num_epochs>1, then EpochCtrlOp is added to the pipeline
        num_iter = 0
        # Here i refers to index, d refers to data element
        for i, d in enumerate(data3.create_tuple_iterator(output_numpy=True, num_epochs=2)):
            num_iter = num_iter + 1
            t = d
            assert i == t[0][0]

        assert num_iter == 10

        with open(PIPELINE_FILE) as f:
            data = json.load(f)
            op_info = data["op_info"]
            assert len(op_info) == 4
            for i in range(4):
                # Note: The following ops are inline ops: Concat, EpochCtrl
                if op_info[i]["op_type"] in ("ConcatOp", "EpochCtrlOp"):
                    # Confirm these inline ops do not have metrics information
                    assert op_info[i]["metrics"] is None
                else:
                    assert "size" in op_info[i]["metrics"]["output_queue"]
                    assert "length" in op_info[i]["metrics"]["output_queue"]
                    assert "throughput" in op_info[i]["metrics"]["output_queue"]

    except Exception as error:
        delete_profiling_files()
        raise error

    else:
        delete_profiling_files()


def test_profiling_inline_ops_pipeline2():
    """
    Test pipeline with many inline ops
    Generator -> Rename -> Skip -> Repeat -> Take
    """
    set_profiling_env_var()

    # In source1 dataset: Number of rows is 10; its values are 0, 1, 2, 3, 4, 5 ... 9
    def source1():
        for i in range(10):
            yield (np.array([i]),)

    data1 = ds.GeneratorDataset(source1, ["col1"])
    data1 = data1.rename(input_columns=["col1"], output_columns=["newcol1"])
    data1 = data1.skip(2)
    data1 = data1.repeat(2)
    data1 = data1.take(12)

    try:
        for _ in data1:
            pass

        with open(PIPELINE_FILE) as f:
            data = json.load(f)
            op_info = data["op_info"]
            assert len(op_info) == 5
            for i in range(5):
                # Check for these inline ops
                if op_info[i]["op_type"] in ("RenameOp", "RepeatOp", "SkipOp", "TakeOp"):
                    # Confirm these inline ops do not have metrics information
                    assert op_info[i]["metrics"] is None
                else:
                    assert "size" in op_info[i]["metrics"]["output_queue"]
                    assert "length" in op_info[i]["metrics"]["output_queue"]
                    assert "throughput" in op_info[i]["metrics"]["output_queue"]

    except Exception as error:
        delete_profiling_files()
        raise error

    else:
        delete_profiling_files()


def test_profiling_sampling_interval():
    """
    Test non-default monitor sampling interval
    """
    set_profiling_env_var()

    interval_origin = ds.config.get_monitor_sampling_interval()

    ds.config.set_monitor_sampling_interval(30)
    interval = ds.config.get_monitor_sampling_interval()
    assert interval == 30

    source = [(np.array([x]),) for x in range(1024)]
    data1 = ds.GeneratorDataset(source, ["data"])
    data1 = data1.shuffle(64)
    data1 = data1.batch(32)

    try:
        for _ in data1:
            pass

    except Exception as error:
        ds.config.set_monitor_sampling_interval(interval_origin)
        delete_profiling_files()
        raise error

    else:
        ds.config.set_monitor_sampling_interval(interval_origin)
        delete_profiling_files()


def test_profiling_basic_pipeline():
    """
    Test with this basic pipeline
    Generator -> Map -> Batch -> Repeat -> EpochCtrl
    """
    set_profiling_env_var()

    def source1():
        for i in range(8000):
            yield (np.array([i]),)

    # Create this basic and common pipeline
    # Leaf/Source-Op -> Map -> Batch -> Repeat
    data1 = ds.GeneratorDataset(source1, ["col1"])

    type_cast_op = C.TypeCast(mstype.int32)
    data1 = data1.map(operations=type_cast_op, input_columns="col1")
    data1 = data1.batch(16)
    data1 = data1.repeat(2)

    try:
        num_iter = 0
        # Note: If create_tuple_iterator() is called with num_epochs>1, then EpochCtrlOp is added to the pipeline
        for _ in data1.create_dict_iterator(num_epochs=2):
            num_iter = num_iter + 1

        assert num_iter == 1000

        with open(PIPELINE_FILE) as f:
            data = json.load(f)
            op_info = data["op_info"]
            assert len(op_info) == 5
            for i in range(5):
                # Check for inline ops
                if op_info[i]["op_type"] in ("EpochCtrlOp", "RepeatOp"):
                    # Confirm these inline ops do not have metrics information
                    assert op_info[i]["metrics"] is None
                else:
                    assert "size" in op_info[i]["metrics"]["output_queue"]
                    assert "length" in op_info[i]["metrics"]["output_queue"]
                    assert "throughput" in op_info[i]["metrics"]["output_queue"]

    except Exception as error:
        delete_profiling_files()
        raise error

    else:
        delete_profiling_files()


def test_profiling_cifar10_pipeline():
    """
    Test with this common pipeline with Cifar10
    Cifar10 -> Map -> Map -> Batch -> Repeat
    """
    set_profiling_env_var()

    # Create this common pipeline
    # Cifar10 -> Map -> Map -> Batch -> Repeat
    DATA_DIR_10 = "../data/dataset/testCifar10Data"
    data1 = ds.Cifar10Dataset(DATA_DIR_10, num_samples=8000)

    type_cast_op = C.TypeCast(mstype.int32)
    data1 = data1.map(operations=type_cast_op, input_columns="label")
    random_horizontal_op = vision.RandomHorizontalFlip()
    data1 = data1.map(operations=random_horizontal_op, input_columns="image")

    data1 = data1.batch(32)
    data1 = data1.repeat(3)

    try:
        num_iter = 0
        # Note: If create_tuple_iterator() is called with num_epochs=1, then EpochCtrlOp is NOT added to the pipeline
        for _ in data1.create_dict_iterator(num_epochs=1):
            num_iter = num_iter + 1

        assert num_iter == 750

        with open(PIPELINE_FILE) as f:
            data = json.load(f)
            op_info = data["op_info"]
            assert len(op_info) == 5
            for i in range(5):
                # Check for inline ops
                if op_info[i]["op_type"] == "RepeatOp":
                    # Confirm these inline ops do not have metrics information
                    assert op_info[i]["metrics"] is None
                else:
                    assert "size" in op_info[i]["metrics"]["output_queue"]
                    assert "length" in op_info[i]["metrics"]["output_queue"]
                    assert "throughput" in op_info[i]["metrics"]["output_queue"]

    except Exception as error:
        delete_profiling_files()
        raise error

    else:
        delete_profiling_files()


if __name__ == "__main__":
    test_profiling_simple_pipeline()
    test_profiling_complex_pipeline()
    test_profiling_inline_ops_pipeline1()
    test_profiling_inline_ops_pipeline2()
    test_profiling_sampling_interval()
    test_profiling_basic_pipeline()
    test_profiling_cifar10_pipeline()
