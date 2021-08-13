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

PIPELINE_FILE = "./pipeline_profiling_0.json"
CPU_UTIL_FILE = "./minddata_cpu_utilization_0.json"
DATASET_ITERATOR_FILE = "./dataset_iterator_profiling_0.txt"
SUMMARY_JSON_FILE = "./minddata_pipeline_summary_0.json"
SUMMARY_CSV_FILE = "./minddata_pipeline_summary_0.csv"
ANALYZE_FILE_PATH = "./"

# This is the minimum subset of expected keys (in alphabetical order) in the MindData Analyzer summary output
EXPECTED_SUMMARY_KEYS = ['avg_cpu_pct', 'children_ids', 'num_workers', 'op_ids', 'op_names', 'parent_id',
                         'per_batch_time', 'pipeline_ops', 'queue_average_size', 'queue_empty_freq_pct',
                         'queue_utilization_pct']


def get_csv_result(file_pathname):
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


def delete_profiling_files():
    """
    Delete the MindData profiling files generated from the test.
    Also disable the MindData Profiling environment variables.
    """
    # Delete MindData profiling files
    os.remove(PIPELINE_FILE)
    os.remove(CPU_UTIL_FILE)
    os.remove(DATASET_ITERATOR_FILE)

    # Delete MindData profiling analyze summary files
    os.remove(SUMMARY_JSON_FILE)
    os.remove(SUMMARY_CSV_FILE)

    # Disable MindData Profiling environment variables
    del os.environ['PROFILING_MODE']
    del os.environ['MINDDATA_PROFILING_DIR']
    del os.environ['DEVICE_ID']


def test_analyze_basic():
    """
    Test MindData profiling analyze summary files exist with basic pipeline.
    Also test basic content (subset of keys and values) from the returned summary result.
    """
    # Confirm MindData Profiling files do not yet exist
    assert os.path.exists(PIPELINE_FILE) is False
    assert os.path.exists(CPU_UTIL_FILE) is False
    assert os.path.exists(DATASET_ITERATOR_FILE) is False
    # Confirm MindData Profiling analyze summary files do not yet exist
    assert os.path.exists(SUMMARY_JSON_FILE) is False
    assert os.path.exists(SUMMARY_CSV_FILE) is False

    # Enable MindData Profiling environment variables
    os.environ['PROFILING_MODE'] = 'true'
    os.environ['MINDDATA_PROFILING_DIR'] = '.'
    os.environ['DEVICE_ID'] = '0'

    def source1():
        for i in range(8000):
            yield (np.array([i]),)

    try:
        # Create this basic and common linear pipeline
        # Generator -> Map -> Batch -> Repeat -> EpochCtrl

        data1 = ds.GeneratorDataset(source1, ["col1"])
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
        assert os.path.exists(PIPELINE_FILE) is True
        assert os.path.exists(CPU_UTIL_FILE) is True
        assert os.path.exists(DATASET_ITERATOR_FILE) is True

        # Call MindData Analyzer for generated MindData profiling files to generate MindData pipeline summary result
        # Note: MindData Analyzer returns the result in 3 formats:
        # 1. returned dictionary
        # 2. JSON file
        # 3. CSV file
        md_analyzer = MinddataProfilingAnalyzer(ANALYZE_FILE_PATH, 0, ANALYZE_FILE_PATH)
        md_summary_dict = md_analyzer.analyze()

        # Confirm MindData Profiling analyze summary files are created
        assert os.path.exists(SUMMARY_JSON_FILE) is True
        assert os.path.exists(SUMMARY_CSV_FILE) is True

        # Build a list of the sorted returned keys
        summary_returned_keys = list(md_summary_dict.keys())
        summary_returned_keys.sort()

        # 1. Confirm expected keys are in returned keys
        for k in EXPECTED_SUMMARY_KEYS:
            assert k in summary_returned_keys

        # Read summary JSON file
        with open(SUMMARY_JSON_FILE) as f:
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
        summary_csv_data = get_csv_result(SUMMARY_CSV_FILE)
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

    except Exception as error:
        delete_profiling_files()
        raise error

    else:
        delete_profiling_files()


if __name__ == "__main__":
    test_analyze_basic()
