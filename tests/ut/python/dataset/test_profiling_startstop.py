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
# ==============================================================================
"""
Test MindData Profiling Start and Stop Support
"""
import json
import os
import numpy as np
import pytest
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore._c_dataengine as cde
import mindspore.dataset.transforms.c_transforms as C

FILES = ["../data/dataset/testTFTestAllTypes/test.data"]
DATASET_ROOT = "../data/dataset/testTFTestAllTypes/"
SCHEMA_FILE = "../data/dataset/testTFTestAllTypes/datasetSchema.json"

# Add file name to rank id mapping so that each profiling file name is unique,
# to support parallel test execution
file_name_map_rank_id = {"test_profiling_early_stop": "0",
                         "test_profiling_delayed_start": "1",
                         "test_profiling_start_start": "2",
                         "test_profiling_multiple_start_stop": "3",
                         "test_profiling_stop_stop": "4",
                         "test_profiling_stop_nostart": "5"}


@pytest.mark.forked
class TestMindDataProfilingStartStop:
    """
    Test MindData Profiling Manager Start-Stop Support
    """

    def setup_class(self):
        """
        Run once for the class
        """
        self._pipeline_file = "./pipeline_profiling"
        self._cpu_util_file = "./minddata_cpu_utilization"
        self._dataset_iterator_file = "./dataset_iterator_profiling"

    def setup_method(self):
        """
        Run before each test function.
        """
        file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
        file_id = file_name_map_rank_id[file_name]

        self.pipeline_file = self._pipeline_file + "_" + file_id + ".json"
        self.cpu_util_file = self._cpu_util_file + "_" + file_id + ".json"
        self.dataset_iterator_file = self._dataset_iterator_file + "_" + file_id + ".txt"

        # Confirm MindData Profiling files do not yet exist
        assert os.path.exists(self.pipeline_file) is False
        assert os.path.exists(self.cpu_util_file) is False
        assert os.path.exists(self.dataset_iterator_file) is False

        # Set the MindData Profiling related environment variables
        os.environ['RANK_ID'] = file_id
        os.environ['DEVICE_ID'] = file_id

    def teardown_method(self):
        """
        Run after each test function.
        """
        # Delete MindData profiling files generated from the test.
        if os.path.exists(self.pipeline_file):
            os.remove(self.pipeline_file)
        if os.path.exists(self.cpu_util_file):
            os.remove(self.cpu_util_file)
        if os.path.exists(self.dataset_iterator_file):
            os.remove(self.dataset_iterator_file)

        # Disable MindData Profiling related environment variables
        del os.environ['RANK_ID']
        del os.environ['DEVICE_ID']

    def confirm_pipeline_file(self, num_ops, op_list=None):
        """
        Confirm pipeline JSON file with <num_ops> in the pipeline and the given optional list of ops
        """
        with open(self.pipeline_file) as file1:
            data = json.load(file1)
            op_info = data["op_info"]
            # Confirm ops in pipeline file
            assert len(op_info) == num_ops
            if op_list:
                for i in range(num_ops):
                    assert op_info[i]["op_type"] in op_list

    def confirm_cpuutil_file(self, num_pipeline_ops):
        """
        Confirm CPU utilization JSON file with <num_pipeline_ops> in the pipeline
        """
        with open(self.cpu_util_file) as file1:
            data = json.load(file1)
            op_info = data["op_info"]
            assert len(op_info) == num_pipeline_ops

    def confirm_dataset_iterator_file(self, num_batches):
        """
        Confirm dataset iterator file exists with the correct number of rows in the file
        """
        assert os.path.exists(self.dataset_iterator_file)
        actual_num_lines = sum(1 for _ in open(self.dataset_iterator_file))
        # Confirm there are 4 lines for each batch in the dataset iterator file
        assert actual_num_lines == 4 * num_batches

    def test_profiling_early_stop(self):
        """
        Test MindData Profiling with Early Stop; profile for some iterations and then stop profiling
        """

        def source1():
            for i in range(8000):
                yield (np.array([i]),)

        # Get instance pointer for MindData profiling manager
        md_profiler = cde.GlobalContext.profiling_manager()

        # Initialize MindData profiling manager
        md_profiler.init()

        # Start MindData Profiling
        md_profiler.start()

        # Create this basic and common pipeline
        # Leaf/Source-Op -> Map -> Batch
        data1 = ds.GeneratorDataset(source1, ["col1"])

        type_cast_op = C.TypeCast(mstype.int32)
        data1 = data1.map(operations=type_cast_op, input_columns="col1")
        data1 = data1.batch(16)

        num_iter = 0
        # Note: If create_dict_iterator() is called with num_epochs>1, then EpochCtrlOp is added to the pipeline
        for _ in data1.create_dict_iterator(num_epochs=2):
            if num_iter == 400:
                # Stop MindData Profiling and Save MindData Profiling Output
                md_profiler.stop()
                md_profiler.save(os.getcwd())

            num_iter += 1

        assert num_iter == 500

        # Confirm the content of the profiling files, including 4 ops in the pipeline JSON file
        self.confirm_pipeline_file(4, ["GeneratorOp", "BatchOp", "MapOp", "EpochCtrlOp"])
        self.confirm_cpuutil_file(4)
        self.confirm_dataset_iterator_file(401)

    def test_profiling_delayed_start(self):
        """
        Test MindData Profiling with Delayed Start; profile for subset of iterations
        """

        def source1():
            for i in range(8000):
                yield (np.array([i]),)

        # Get instance pointer for MindData profiling manager
        md_profiler = cde.GlobalContext.profiling_manager()

        # Initialize MindData profiling manager
        md_profiler.init()

        # Create this basic and common pipeline
        # Leaf/Source-Op -> Map -> Batch
        data1 = ds.GeneratorDataset(source1, ["col1"])

        type_cast_op = C.TypeCast(mstype.int32)
        data1 = data1.map(operations=type_cast_op, input_columns="col1")
        data1 = data1.batch(16)

        num_iter = 0
        # Note: If create_dict_iterator() is called with num_epochs=1, then EpochCtrlOp is not added to the pipeline
        for _ in data1.create_dict_iterator(num_epochs=1):
            if num_iter == 5:
                # Start MindData Profiling
                md_profiler.start()
            elif num_iter == 400:
                # Stop MindData Profiling and Save MindData Profiling Output
                md_profiler.stop()
                md_profiler.save(os.getcwd())

            num_iter += 1

        assert num_iter == 500

        # Confirm the content of the profiling files, including 3 ops in the pipeline JSON file
        self.confirm_pipeline_file(3, ["GeneratorOp", "BatchOp", "MapOp"])
        self.confirm_cpuutil_file(3)
        self.confirm_dataset_iterator_file(395)

    def test_profiling_multiple_start_stop(self):
        """
        Test MindData Profiling with Delayed Start and Multiple Start-Stop Sequences
        """

        def source1():
            for i in range(8000):
                yield (np.array([i]),)

        # Get instance pointer for MindData profiling manager
        md_profiler = cde.GlobalContext.profiling_manager()

        # Initialize MindData profiling manager
        md_profiler.init()

        # Create this basic and common pipeline
        # Leaf/Source-Op -> Map -> Batch
        data1 = ds.GeneratorDataset(source1, ["col1"])

        type_cast_op = C.TypeCast(mstype.int32)
        data1 = data1.map(operations=type_cast_op, input_columns="col1")
        data1 = data1.batch(16)

        num_iter = 0
        # Note: If create_dict_iterator() is called with num_epochs=1, then EpochCtrlOp is not added to the pipeline
        for _ in data1.create_dict_iterator(num_epochs=1):
            if num_iter == 5:
                # Start MindData Profiling
                md_profiler.start()
            elif num_iter == 40:
                # Stop MindData Profiling
                md_profiler.stop()
            if num_iter == 200:
                # Start MindData Profiling
                md_profiler.start()
            elif num_iter == 400:
                # Stop MindData Profiling
                md_profiler.stop()

            num_iter += 1

        # Save MindData Profiling Output
        md_profiler.save(os.getcwd())
        assert num_iter == 500

        # Confirm the content of the profiling files, including 3 ops in the pipeline JSON file
        self.confirm_pipeline_file(3, ["GeneratorOp", "BatchOp", "MapOp"])
        self.confirm_cpuutil_file(3)
        # Note: The dataset iterator file should only contain data for batches 200 to 400
        self.confirm_dataset_iterator_file(200)

    def test_profiling_start_start(self):
        """
        Test MindData Profiling with Start followed by Start - user error scenario
        """
        # Get instance pointer for MindData profiling manager
        md_profiler = cde.GlobalContext.profiling_manager()

        # Initialize MindData profiling manager
        md_profiler.init()

        # Start MindData Profiling
        md_profiler.start()

        with pytest.raises(RuntimeError) as info:
            # Reissue Start MindData Profiling
            md_profiler.start()

        assert "MD ProfilingManager is already running." in str(info)

        # Stop MindData Profiling
        md_profiler.stop()

    def test_profiling_stop_stop(self):
        """
        Test MindData Profiling with Stop followed by Stop - user warning scenario
        """
        # Get instance pointer for MindData profiling manager
        md_profiler = cde.GlobalContext.profiling_manager()

        # Initialize MindData profiling manager
        md_profiler.init()

        # Start MindData Profiling
        md_profiler.start()

        # Stop MindData Profiling and Save MindData Profiling Output
        md_profiler.stop()
        md_profiler.save(os.getcwd())

        # Reissue Stop MindData Profiling
        # A warning "MD ProfilingManager had already stopped" is produced.
        md_profiler.stop()

    def test_profiling_stop_nostart(self):
        """
        Test MindData Profiling with Stop not without prior Start - user error scenario
        """
        # Get instance pointer for MindData profiling manager
        md_profiler = cde.GlobalContext.profiling_manager()

        # Initialize MindData profiling manager
        md_profiler.init()

        with pytest.raises(RuntimeError) as info:
            # Stop MindData Profiling - without prior Start()
            md_profiler.stop()

        assert "MD ProfilingManager has not started yet." in str(info)

        # Start MindData Profiling
        md_profiler.start()
        # Stop MindData Profiling - to return profiler to a healthy state
        md_profiler.stop()
