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
Test MindData Profiling Analyzer Support
"""
import csv
import json
import os
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
from mindspore.profiler.parser.minddata_analyzer import MinddataProfilingAnalyzer


class TestMinddataProfilingAnalyzer():
    """
    Test the MinddataProfilingAnalyzer class
    """

    def setup_class(self):
        """
        Run once for the class
        """
        # Define filenames and path used for the MinddataProfilingAnalyzer tests. Use device_id=7.
        self._PIPELINE_FILE = "./pipeline_profiling_7.json"
        self._CPU_UTIL_FILE = "./minddata_cpu_utilization_7.json"
        self._DATASET_ITERATOR_FILE = "./dataset_iterator_profiling_7.txt"
        self._SUMMARY_JSON_FILE = "./minddata_pipeline_summary_7.json"
        self._SUMMARY_CSV_FILE = "./minddata_pipeline_summary_7.csv"
        self._ANALYZE_FILE_PATH = "./"

        # This is the set of keys for success case
        self._EXPECTED_SUMMARY_KEYS_SUCCESS = \
            ['avg_cpu_pct', 'avg_cpu_pct_per_worker', 'children_ids', 'num_workers', 'op_ids', 'op_names',
             'parent_id', 'per_batch_time', 'per_pipeline_time', 'per_push_queue_time', 'pipeline_ops',
             'queue_average_size', 'queue_empty_freq_pct', 'queue_utilization_pct']


    def setup_method(self):
        """
        Run before each test function.
        """
        # Confirm MindData Profiling files do not yet exist
        assert os.path.exists(self._PIPELINE_FILE) is False
        assert os.path.exists(self._CPU_UTIL_FILE) is False
        assert os.path.exists(self._DATASET_ITERATOR_FILE) is False
        # Confirm MindData Profiling analyze summary files do not yet exist
        assert os.path.exists(self._SUMMARY_JSON_FILE) is False
        assert os.path.exists(self._SUMMARY_CSV_FILE) is False

        # Set the MindData Profiling environment variables
        os.environ['PROFILING_MODE'] = 'true'
        os.environ['MINDDATA_PROFILING_DIR'] = '.'
        os.environ['RANK_ID'] = '7'


    def teardown_method(self):
        """
        Run after each test function.
        """
        # Delete MindData profiling files generated from the test.
        os.remove(self._PIPELINE_FILE)
        os.remove(self._CPU_UTIL_FILE)
        os.remove(self._DATASET_ITERATOR_FILE)

        # Delete MindData profiling analyze summary files generated from the test.
        os.remove(self._SUMMARY_JSON_FILE)
        os.remove(self._SUMMARY_CSV_FILE)

        # Disable MindData Profiling environment variables
        del os.environ['PROFILING_MODE']
        del os.environ['MINDDATA_PROFILING_DIR']
        del os.environ['RANK_ID']


    def get_csv_result(self, file_pathname):
        """
        Get result from the CSV file.

        Args:
            file_pathname (str): The CSV file pathname.

        Returns:
            list[list], the parsed CSV information.
        """
        result = []
        with open(file_pathname, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                result.append(row)
        return result


    def verify_md_summary(self, md_summary_dict, EXPECTED_SUMMARY_KEYS):
        """
        Verify the content of the 3 variations of the MindData Profiling analyze summary output.
        """

        # Confirm MindData Profiling analyze summary files are created
        assert os.path.exists(self._SUMMARY_JSON_FILE) is True
        assert os.path.exists(self._SUMMARY_CSV_FILE) is True

        # Build a list of the sorted returned keys
        summary_returned_keys = list(md_summary_dict.keys())
        summary_returned_keys.sort()

        # 1. Confirm expected keys are in returned keys
        for k in EXPECTED_SUMMARY_KEYS:
            assert k in summary_returned_keys

        # Read summary JSON file
        with open(self._SUMMARY_JSON_FILE) as f:
            summary_json_data = json.load(f)
        # Build a list of the sorted JSON keys
        summary_json_keys = list(summary_json_data.keys())
        summary_json_keys.sort()

        # 2a. Confirm expected keys are in JSON file keys
        for k in EXPECTED_SUMMARY_KEYS:
            assert k in summary_json_keys

        # 2b. Confirm returned dictionary keys are identical to JSON file keys
        np.testing.assert_array_equal(summary_returned_keys, summary_json_keys)

        # Read summary CSV file
        summary_csv_data = self.get_csv_result(self._SUMMARY_CSV_FILE)
        # Build a list of the sorted CSV keys from the first column in the CSV file
        summary_csv_keys = []
        for x in summary_csv_data:
            summary_csv_keys.append(x[0])
        summary_csv_keys.sort()

        # 3a. Confirm expected keys are in the first column of the CSV file
        for k in EXPECTED_SUMMARY_KEYS:
            assert k in summary_csv_keys

        # 3b. Confirm returned dictionary keys are identical to CSV file first column keys
        np.testing.assert_array_equal(summary_returned_keys, summary_csv_keys)


    def mysource(self):
        """Source for data values"""
        for i in range(8000):
            yield (np.array([i]),)


    def test_analyze_basic(self):
        """
        Test MindData profiling analyze summary files exist with basic pipeline.
        Also test basic content (subset of keys and values) from the returned summary result.
        """
        # Create this basic and common linear pipeline
        # Generator -> Map -> Batch -> Repeat -> EpochCtrl
        data1 = ds.GeneratorDataset(self.mysource, ["col1"])
        type_cast_op = C.TypeCast(mstype.int32)
        data1 = data1.map(operations=type_cast_op, input_columns="col1")
        data1 = data1.batch(16)
        data1 = data1.repeat(2)

        num_iter = 0
        # Note: If create_tuple_iterator() is called with num_epochs>1, then EpochCtrlOp is added to the pipeline
        for _ in data1.create_dict_iterator(num_epochs=2):
            num_iter = num_iter + 1

        # Confirm number of rows returned
        assert num_iter == 1000

        # Confirm MindData Profiling files are created
        assert os.path.exists(self._PIPELINE_FILE) is True
        assert os.path.exists(self._CPU_UTIL_FILE) is True
        assert os.path.exists(self._DATASET_ITERATOR_FILE) is True

        # Call MindData Analyzer for generated MindData profiling files to generate MindData pipeline summary result
        md_analyzer = MinddataProfilingAnalyzer(self._ANALYZE_FILE_PATH, 7, self._ANALYZE_FILE_PATH)
        md_summary_dict = md_analyzer.analyze()

        # Verify MindData Profiling Analyze Summary output
        # Note: MindData Analyzer returns the result in 3 formats:
        # 1. returned dictionary
        # 2. JSON file
        # 3. CSV file
        self.verify_md_summary(md_summary_dict, self._EXPECTED_SUMMARY_KEYS_SUCCESS)

        # 4. Verify non-variant values or number of values in the tested pipeline for certain keys
        # of the returned dictionary
        # Note: Values of num_workers are not tested since default may change in the future
        # Note: Values related to queue metrics are not tested since they may vary on different execution environments
        assert md_summary_dict["pipeline_ops"] == ["EpochCtrl(id=0)", "Repeat(id=1)", "Batch(id=2)", "Map(id=3)",
                                                   "Generator(id=4)"]
        assert md_summary_dict["op_names"] == ["EpochCtrl", "Repeat", "Batch", "Map", "Generator"]
        assert md_summary_dict["op_ids"] == [0, 1, 2, 3, 4]
        assert len(md_summary_dict["num_workers"]) == 5
        assert len(md_summary_dict["queue_average_size"]) == 5
        assert len(md_summary_dict["queue_utilization_pct"]) == 5
        assert len(md_summary_dict["queue_empty_freq_pct"]) == 5
        assert md_summary_dict["children_ids"] == [[1], [2], [3], [4], []]
        assert md_summary_dict["parent_id"] == [-1, 0, 1, 2, 3]
        assert len(md_summary_dict["avg_cpu_pct"]) == 5


    def test_analyze_sequential_pipelines_invalid(self):
        """
        Test invalid scenario in which MinddataProfilingAnalyzer is called for two sequential pipelines.
        """
        # Create the pipeline
        # Generator -> Map -> Batch -> EpochCtrl
        data1 = ds.GeneratorDataset(self.mysource, ["col1"])
        type_cast_op = C.TypeCast(mstype.int32)
        data1 = data1.map(operations=type_cast_op, input_columns="col1")
        data1 = data1.batch(64)

        # Phase 1 - For the pipeline, call create_tuple_iterator with num_epochs>1
        # Note: This pipeline has 4 ops: Generator -> Map -> Batch -> EpochCtrl
        num_iter = 0
        # Note: If create_tuple_iterator() is called with num_epochs>1, then EpochCtrlOp is added to the pipeline
        for _ in data1.create_dict_iterator(num_epochs=2):
            num_iter = num_iter + 1

        # Confirm number of rows returned
        assert num_iter == 125

        # Confirm MindData Profiling files are created
        assert os.path.exists(self._PIPELINE_FILE) is True
        assert os.path.exists(self._CPU_UTIL_FILE) is True
        assert os.path.exists(self._DATASET_ITERATOR_FILE) is True

        # Phase 2 - For the pipeline, call create_tuple_iterator with num_epochs=1
        # Note: This pipeline has 3 ops: Generator -> Map -> Batch
        num_iter = 0
        # Note: If create_tuple_iterator() is called with num_epochs=1, then EpochCtrlOp is NOT added to the pipeline
        for _ in data1.create_dict_iterator(num_epochs=1):
            num_iter = num_iter + 1

        # Confirm number of rows returned
        assert num_iter == 125

        # Confirm MindData Profiling files are created
        # Note: There is an MD bug in which which the pipeline file is not recreated;
        #       it still has 4 ops instead of 3 ops
        assert os.path.exists(self._PIPELINE_FILE) is True
        assert os.path.exists(self._CPU_UTIL_FILE) is True
        assert os.path.exists(self._DATASET_ITERATOR_FILE) is True

        # Call MindData Analyzer for generated MindData profiling files to generate MindData pipeline summary result
        md_analyzer = MinddataProfilingAnalyzer(self._ANALYZE_FILE_PATH, 7, self._ANALYZE_FILE_PATH)
        md_summary_dict = md_analyzer.analyze()

        # Verify MindData Profiling Analyze Summary output
        self.verify_md_summary(md_summary_dict, self._EXPECTED_SUMMARY_KEYS_SUCCESS)

        # Confirm pipeline data contains info for 3 ops
        assert md_summary_dict["pipeline_ops"] == ["Batch(id=0)", "Map(id=1)", "Generator(id=2)"]

        # Verify CPU util data contains info for 3 ops
        assert len(md_summary_dict["avg_cpu_pct"]) == 3
