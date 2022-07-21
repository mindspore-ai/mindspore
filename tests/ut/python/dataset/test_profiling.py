# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
import pytest
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as vision
import mindspore._c_dataengine as cde

FILES = ["../data/dataset/testTFTestAllTypes/test.data"]
DATASET_ROOT = "../data/dataset/testTFTestAllTypes/"
SCHEMA_FILE = "../data/dataset/testTFTestAllTypes/datasetSchema.json"


@pytest.mark.forked
class TestMinddataProfilingManager:
    """
    Test MinddataProfilingManager
    Note: Use pytest fixture tmp_path to create files within this temporary directory,
    which is automatically created for each test and deleted at the end of the test.
    """

    def setup_class(self):
        """
        Run once for the class
        """
        # Get instance pointer for MindData profiling manager
        self.md_profiler = cde.GlobalContext.profiling_manager()

    def setup_method(self):
        """
        Run before each test function.
        """

        # Set the MindData Profiling related environment variables
        os.environ['RANK_ID'] = "0"
        os.environ['DEVICE_ID'] = "0"

        # Initialize MindData profiling manager
        self.md_profiler.init()

        # Start MindData Profiling
        self.md_profiler.start()

    def teardown_method(self):
        """
        Run after each test function.
        """

        # Disable MindData Profiling related environment variables
        del os.environ['RANK_ID']
        del os.environ['DEVICE_ID']

    def confirm_cpuutil(self, cpu_util_file, num_pipeline_ops):
        """
        Confirm CPU utilization JSON file with <num_pipeline_ops> in the pipeline
        """
        with open(cpu_util_file) as file1:
            data = json.load(file1)
            op_info = data["op_info"]
            assert len(op_info) == num_pipeline_ops

            # Confirm memory fields exist
            assert "pss_mbytes" in data["process_memory_info"]
            assert "rss_mbytes" in data["process_memory_info"]
            assert "vss_mbytes" in data["process_memory_info"]
            assert "available_sys_memory_mbytes" in data["system_memory_info"]
            assert "total_sys_memory_mbytes" in data["system_memory_info"]
            assert "used_sys_memory_mbytes" in data["system_memory_info"]

            # Perform sanity check on memory information
            assert data["process_memory_info"]["pss_mbytes"][0] > 0
            assert data["process_memory_info"]["rss_mbytes"][0] > 0
            assert data["process_memory_info"]["vss_mbytes"][0] > 0
            assert data["system_memory_info"]["available_sys_memory_mbytes"][0] > 0
            assert data["system_memory_info"]["total_sys_memory_mbytes"][0] > 0
            assert data["system_memory_info"]["used_sys_memory_mbytes"][0] > 0

    def confirm_ops_in_pipeline(self, pipeline_file, num_ops, op_list):
        """
        Confirm pipeline JSON file with <num_ops> are in the pipeline and the given list of ops
        """
        with open(pipeline_file) as file1:
            data = json.load(file1)
            op_info = data["op_info"]
            # Confirm ops in pipeline file
            assert len(op_info) == num_ops
            for i in range(num_ops):
                assert op_info[i]["op_type"] in op_list

    def confirm_dataset_iterator_file(self, dataset_iterator_file, num_batches):
        """
        Confirm dataset iterator file exists with the correct number of rows in the file
        """
        assert os.path.exists(dataset_iterator_file)
        actual_num_lines = sum(1 for _ in open(dataset_iterator_file))
        # Confirm there are 4 lines for each batch in the dataset iterator file
        assert actual_num_lines == 4 * num_batches

    def test_profiling_simple_pipeline(self, tmp_path):
        """
        Feature: MindData Profiling Manager
        Description: Test MindData profiling simple pipeline (Generator -> Shuffle -> Batch)
        Expectation: Runs successfully
        """
        source = [(np.array([x]),) for x in range(1024)]
        data1 = ds.GeneratorDataset(source, ["data"])
        data1 = data1.shuffle(64)
        data1 = data1.batch(32)
        # Check output shape type and dataset size
        assert data1.output_shapes() == [[32, 1]]
        assert [str(tp) for tp in data1.output_types()] == ["int64"]
        assert data1.get_dataset_size() == 32

        # Stop MindData Profiling and save output files to tmp_path
        self.md_profiler.stop()
        self.md_profiler.save(str(tmp_path))

        pipeline_file = str(tmp_path) + "/pipeline_profiling_0.json"
        cpu_util_file = str(tmp_path) + "/minddata_cpu_utilization_0.json"
        dataset_iterator_file = str(tmp_path) + "/dataset_iterator_profiling_0.txt"

        # Confirm no profiling files are produced (since no MindData pipeline has been executed)
        assert os.path.exists(pipeline_file) is False
        assert os.path.exists(cpu_util_file) is False
        assert os.path.exists(dataset_iterator_file) is False

        # Start MindData Profiling
        self.md_profiler.start()

        # Execute MindData Pipeline
        for _ in data1:
            pass

        # Stop MindData Profiling and save output files to tmp_path
        self.md_profiler.stop()
        self.md_profiler.save(str(tmp_path))

        # Confirm profiling files now exist
        assert os.path.exists(pipeline_file) is True
        assert os.path.exists(cpu_util_file) is True
        assert os.path.exists(dataset_iterator_file) is True

    def test_profiling_complex_pipeline(self, tmp_path):
        """
        Feature: MindData Profiling Manager
        Description: Test MindData profiling complex pipeline:

        Generator -> Map     ->
                                 -> Zip
        TFReader  -> Shuffle ->

        Expectation: Runs successfully
        """
        source = [(np.array([x]),) for x in range(1024)]
        data1 = ds.GeneratorDataset(source, ["gen"])
        data1 = data1.map(operations=[(lambda x: x + 1)], input_columns=["gen"])

        pattern = DATASET_ROOT + "/test.data"
        data2 = ds.TFRecordDataset(pattern, SCHEMA_FILE, shuffle=ds.Shuffle.FILES)
        data2 = data2.shuffle(4)

        data3 = ds.zip((data1, data2))

        for _ in data3:
            pass

        # Stop MindData Profiling and save output files to tmp_path
        self.md_profiler.stop()
        self.md_profiler.save(str(tmp_path))

        pipeline_file = str(tmp_path) + "/pipeline_profiling_0.json"
        cpu_util_file = str(tmp_path) + "/minddata_cpu_utilization_0.json"
        dataset_iterator_file = str(tmp_path) + "/dataset_iterator_profiling_0.txt"

        with open(pipeline_file) as f:
            data = json.load(f)
            op_info = data["op_info"]
            assert len(op_info) == 5
            for i in range(5):
                if op_info[i]["op_type"] != "ZipOp":
                    assert "size" in op_info[i]["metrics"]["output_queue"]
                    assert "length" in op_info[i]["metrics"]["output_queue"]
                else:
                    # Note: Zip is an inline op and hence does not have metrics information
                    assert op_info[i]["metrics"] is None

        # Confirm CPU util JSON file content, when 5 ops are in the pipeline JSON file
        self.confirm_cpuutil(cpu_util_file, 5)

        # Confirm dataset iterator file content
        self.confirm_dataset_iterator_file(dataset_iterator_file, 12)

    def test_profiling_inline_ops_pipeline1(self, tmp_path):
        """
        Feature: MindData Profiling Manager
        Description: Test MindData profiling pipeline with inline ops (Concat and EpochCtrl):

        Generator ->
                     Concat -> EpochCtrl
        Generator ->

        Expectation: Runs successfully
        """

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

        num_iter = 0
        # Note: set num_epochs=2 in create_tuple_iterator(), so that EpochCtrl op is added to the pipeline
        # Here i refers to index, d refers to data element
        for i, d in enumerate(data3.create_tuple_iterator(num_epochs=2, output_numpy=True)):
            num_iter += 1
            t = d
            assert i == t[0][0]

        assert num_iter == 10

        # Stop MindData Profiling and save output files to tmp_path
        self.md_profiler.stop()
        self.md_profiler.save(str(tmp_path))

        pipeline_file = str(tmp_path) + "/pipeline_profiling_0.json"
        cpu_util_file = str(tmp_path) + "/minddata_cpu_utilization_0.json"
        dataset_iterator_file = str(tmp_path) + "/dataset_iterator_profiling_0.txt"

        # Confirm pipeline is created with EpochCtrl op
        with open(pipeline_file) as f:
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

        # Confirm CPU util JSON file content, when 4 ops are in the pipeline JSON file
        self.confirm_cpuutil(cpu_util_file, 4)

        # Confirm dataset iterator file content
        self.confirm_dataset_iterator_file(dataset_iterator_file, 10)

    def test_profiling_inline_ops_pipeline2(self, tmp_path):
        """
        Feature: MindData Profiling Manager
        Description: Test MindData profiling pipeline with many inline ops
            (Generator -> Rename -> Skip -> Repeat -> Take)
        Expectation: Runs successfully
        """

        # In source1 dataset: Number of rows is 10; its values are 0, 1, 2, 3, 4, 5 ... 9
        def source1():
            for i in range(10):
                yield (np.array([i]),)

        data1 = ds.GeneratorDataset(source1, ["col1"])
        data1 = data1.rename(input_columns=["col1"], output_columns=["newcol1"])
        data1 = data1.skip(2)
        data1 = data1.repeat(2)
        data1 = data1.take(12)

        for _ in data1:
            pass

        # Stop MindData Profiling and save output files to tmp_path
        self.md_profiler.stop()
        self.md_profiler.save(str(tmp_path))

        pipeline_file = str(tmp_path) + "/pipeline_profiling_0.json"
        cpu_util_file = str(tmp_path) + "/minddata_cpu_utilization_0.json"
        dataset_iterator_file = str(tmp_path) + "/dataset_iterator_profiling_0.txt"

        with open(pipeline_file) as f:
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

        # Confirm CPU util JSON file content, when 5 ops are in the pipeline JSON file
        self.confirm_cpuutil(cpu_util_file, 5)

        # Confirm dataset iterator file content
        self.confirm_dataset_iterator_file(dataset_iterator_file, 12)

    def test_profiling_sampling_interval(self, tmp_path):
        """
        Feature: MindData Profiling Manager
        Description: Test non-default monitor sampling interval
        Expectation: Runs successfully
        """
        interval_origin = ds.config.get_monitor_sampling_interval()

        ds.config.set_monitor_sampling_interval(30)
        interval = ds.config.get_monitor_sampling_interval()
        assert interval == 30

        source = [(np.array([x]),) for x in range(1024)]
        data1 = ds.GeneratorDataset(source, ["data"])
        data1 = data1.shuffle(64)
        data1 = data1.batch(32)

        for _ in data1:
            pass

        ds.config.set_monitor_sampling_interval(interval_origin)

        # Stop MindData Profiling and save output files to tmp_path
        self.md_profiler.stop()
        self.md_profiler.save(str(tmp_path))

        pipeline_file = str(tmp_path) + "/pipeline_profiling_0.json"
        cpu_util_file = str(tmp_path) + "/minddata_cpu_utilization_0.json"
        dataset_iterator_file = str(tmp_path) + "/dataset_iterator_profiling_0.txt"

        # Confirm pipeline file and CPU util file each have 3 ops
        self.confirm_ops_in_pipeline(pipeline_file, 3, ["GeneratorOp", "BatchOp", "ShuffleOp"])
        self.confirm_cpuutil(cpu_util_file, 3)

        # Confirm dataset iterator file content
        self.confirm_dataset_iterator_file(dataset_iterator_file, 32)

    def test_profiling_basic_pipeline(self, tmp_path):
        """
        Feature: MindData Profiling Manager
        Description: Test MindData profiling pipeline with basic pipeline
            (Generator -> Map -> Batch -> Repeat -> EpochCtrl)
        Expectation: Runs successfully
        """

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

        num_iter = 0
        # Note: If create_dict_iterator() is called with num_epochs>1, then EpochCtrlOp is added to the pipeline
        for _ in data1.create_dict_iterator(num_epochs=2):
            num_iter += 1

        assert num_iter == 1000

        # Stop MindData Profiling and save output files to tmp_path
        self.md_profiler.stop()
        self.md_profiler.save(str(tmp_path))

        pipeline_file = str(tmp_path) + "/pipeline_profiling_0.json"
        cpu_util_file = str(tmp_path) + "/minddata_cpu_utilization_0.json"
        dataset_iterator_file = str(tmp_path) + "/dataset_iterator_profiling_0.txt"

        with open(pipeline_file) as f:
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

        # Confirm CPU util JSON file content, when 5 ops are in the pipeline JSON file
        self.confirm_cpuutil(cpu_util_file, 5)

        # Confirm dataset iterator file content
        self.confirm_dataset_iterator_file(dataset_iterator_file, 1000)

    def test_profiling_cifar10_pipeline(self, tmp_path):
        """
        Feature: MindData Profiling Manager
        Description: Test MindData profiling with common pipeline with Cifar10
            (Cifar10 -> Map -> Map -> Batch -> Repeat)
        Expectation: Runs successfully
        """
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

        num_iter = 0
        # Note: If create_dict_iterator() is called with num_epochs=1, then EpochCtrlOp is NOT added to the pipeline
        for _ in data1.create_dict_iterator(num_epochs=1):
            num_iter += 1

        assert num_iter == 750

        # Stop MindData Profiling and save output files to tmp_path
        self.md_profiler.stop()
        self.md_profiler.save(str(tmp_path))

        pipeline_file = str(tmp_path) + "/pipeline_profiling_0.json"
        cpu_util_file = str(tmp_path) + "/minddata_cpu_utilization_0.json"
        dataset_iterator_file = str(tmp_path) + "/dataset_iterator_profiling_0.txt"

        with open(pipeline_file) as f:
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

        # Confirm CPU util JSON file content, when 5 ops are in the pipeline JSON file
        self.confirm_cpuutil(cpu_util_file, 5)

        # Confirm dataset iterator file content
        self.confirm_dataset_iterator_file(dataset_iterator_file, 750)

    def test_profiling_seq_pipelines_epochctrl3(self, tmp_path):
        """
        Feature: MindData Profiling Manager
        Description: Test MindData profiling with these 2 sequential pipelines
            1) Generator -> Batch -> EpochCtrl
            2) Generator -> Batch
            Note: This is a simplification of the user scenario to use the same pipeline for train and then eval
        Expectation: Runs successfully
        """
        source = [(np.array([x]),) for x in range(64)]
        data1 = ds.GeneratorDataset(source, ["data"])
        data1 = data1.batch(32)

        # Test A - Call create_dict_iterator with num_epochs>1
        num_iter = 0
        # Note: If create_dict_iterator() is called with num_epochs>1, then EpochCtrlOp is added to the pipeline
        for _ in data1.create_dict_iterator(num_epochs=2):
            num_iter += 1
        assert num_iter == 2

        # Stop MindData Profiling and save output files to tmp_path
        self.md_profiler.stop()
        self.md_profiler.save(str(tmp_path))

        pipeline_file = str(tmp_path) + "/pipeline_profiling_0.json"
        cpu_util_file = str(tmp_path) + "/minddata_cpu_utilization_0.json"
        dataset_iterator_file = str(tmp_path) + "/dataset_iterator_profiling_0.txt"

        # Confirm pipeline file and CPU util file each have 3 ops
        self.confirm_ops_in_pipeline(pipeline_file, 3, ["GeneratorOp", "BatchOp", "EpochCtrlOp"])
        self.confirm_cpuutil(cpu_util_file, 3)

        # Test B - Call create_dict_iterator with num_epochs=1

        # Initialize and Start MindData profiling manager
        self.md_profiler.init()
        self.md_profiler.start()

        num_iter = 0
        # Note: If create_dict_iterator() is called with num_epochs=1,
        # then EpochCtrlOp should not be NOT added to the pipeline
        for _ in data1.create_dict_iterator(num_epochs=1):
            num_iter += 1
        assert num_iter == 2

        # Stop MindData Profiling and save output files to tmp_path
        self.md_profiler.stop()
        self.md_profiler.save(str(tmp_path))

        # Confirm pipeline file and CPU util file each have 2 ops
        self.confirm_ops_in_pipeline(pipeline_file, 2, ["GeneratorOp", "BatchOp"])
        self.confirm_cpuutil(cpu_util_file, 2)

        # Confirm dataset iterator file content
        self.confirm_dataset_iterator_file(dataset_iterator_file, 2)

    def test_profiling_seq_pipelines_epochctrl2(self, tmp_path):
        """
        Feature: MindData Profiling Manager
        Description: Test MindData profiling with these 2 sequential pipelines
            1) Generator -> Batch
            2) Generator -> Batch -> EpochCtrl
        Expectation: Runs successfully
        """
        source = [(np.array([x]),) for x in range(64)]
        data2 = ds.GeneratorDataset(source, ["data"])
        data2 = data2.batch(16)

        # Test A - Call create_dict_iterator with num_epochs=1
        num_iter = 0
        # Note: If create_dict_iterator() is called with num_epochs=1, then EpochCtrlOp is NOT added to the pipeline
        for _ in data2.create_dict_iterator(num_epochs=1):
            num_iter += 1
        assert num_iter == 4

        # Stop MindData Profiling and save output files to tmp_path
        self.md_profiler.stop()
        self.md_profiler.save(str(tmp_path))

        pipeline_file = str(tmp_path) + "/pipeline_profiling_0.json"
        cpu_util_file = str(tmp_path) + "/minddata_cpu_utilization_0.json"
        dataset_iterator_file = str(tmp_path) + "/dataset_iterator_profiling_0.txt"

        # Confirm pipeline file and CPU util file each have 2 ops
        self.confirm_ops_in_pipeline(pipeline_file, 2, ["GeneratorOp", "BatchOp"])
        self.confirm_cpuutil(cpu_util_file, 2)

        # Test B - Call create_dict_iterator with num_epochs>1

        # Initialize and Start MindData profiling manager
        self.md_profiler.init()
        self.md_profiler.start()

        num_iter = 0
        # Note: If create_dict_iterator() is called with num_epochs>1,
        # then EpochCtrlOp should be added to the pipeline
        for _ in data2.create_dict_iterator(num_epochs=2):
            num_iter += 1
        assert num_iter == 4

        # Stop MindData Profiling and save output files to tmp_path
        self.md_profiler.stop()
        self.md_profiler.save(str(tmp_path))

        # Confirm pipeline file and CPU util file each have 3 ops
        self.confirm_ops_in_pipeline(pipeline_file, 3, ["GeneratorOp", "BatchOp", "EpochCtrlOp"])
        self.confirm_cpuutil(cpu_util_file, 3)

        # Confirm dataset iterator file content
        self.confirm_dataset_iterator_file(dataset_iterator_file, 4)

    def test_profiling_seq_pipelines_repeat(self, tmp_path):
        """
        Feature: MindData Profiling Manager
        Description: Test MindData profiling with these 2 sequential pipelines
            1) Generator -> Batch
            2) Generator -> Batch -> Repeat
        Expectation: Runs successfully
        """
        source = [(np.array([x]),) for x in range(64)]
        data2 = ds.GeneratorDataset(source, ["data"])
        data2 = data2.batch(16)

        # Test A - Call create_dict_iterator with 2 ops in pipeline
        num_iter = 0
        for _ in data2.create_dict_iterator(num_epochs=1):
            num_iter += 1
        assert num_iter == 4

        # Stop MindData Profiling and save output files to tmp_path
        self.md_profiler.stop()
        self.md_profiler.save(str(tmp_path))

        pipeline_file = str(tmp_path) + "/pipeline_profiling_0.json"
        cpu_util_file = str(tmp_path) + "/minddata_cpu_utilization_0.json"
        dataset_iterator_file = str(tmp_path) + "/dataset_iterator_profiling_0.txt"

        # Confirm pipeline file and CPU util file each have 2 ops
        self.confirm_ops_in_pipeline(pipeline_file, 2, ["GeneratorOp", "BatchOp"])
        self.confirm_cpuutil(cpu_util_file, 2)

        # Test B - Add repeat op to pipeline.  Call create_dict_iterator with 3 ops in pipeline

        # Initialize and Start MindData profiling manager
        self.md_profiler.init()
        self.md_profiler.start()

        data2 = data2.repeat(5)
        num_iter = 0
        for _ in data2.create_dict_iterator(num_epochs=1):
            num_iter += 1
        assert num_iter == 20

        # Stop MindData Profiling and save output files to tmp_path
        self.md_profiler.stop()
        self.md_profiler.save(str(tmp_path))

        # Confirm pipeline file and CPU util file each have 3 ops
        self.confirm_ops_in_pipeline(pipeline_file, 3, ["GeneratorOp", "BatchOp", "RepeatOp"])
        self.confirm_cpuutil(cpu_util_file, 3)

        # Confirm dataset iterator file content
        self.confirm_dataset_iterator_file(dataset_iterator_file, 20)
