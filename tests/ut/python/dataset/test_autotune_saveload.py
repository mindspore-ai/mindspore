# Copyright 2022 Huawei Technologies Co., Ltd
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
Test Dataset AutoTune's Save and Load Configuration support
"""
import os
import json

DATA_DIR = "../data/dataset/testPK/data"


def data_pipeline_same(file1, file2):
    assert file1.exists()
    assert file2.exists()
    with file1.open() as f1, file2.open() as f2:
        pipeline1 = json.load(f1)
        pipeline1 = pipeline1["tree"] if "tree" in pipeline1 else pipeline1
        pipeline2 = json.load(f2)
        pipeline2 = pipeline2["tree"] if "tree" in pipeline2 else pipeline2
        return pipeline1 == pipeline2


def validate_jsonfile(filepath):
    try:
        file_exist = os.path.exists(filepath)
        with open(filepath, 'r', encoding="utf-8") as jfile:
            loaded_json = json.load(jfile)
    except IOError:
        return False
    return file_exist and isinstance(loaded_json, dict)
