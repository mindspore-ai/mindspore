# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
Test mindrecord
"""
import os
import pytest
from mindspore.mindrecord import FileWriter

from tests.mark_utils import arg_mark

@arg_mark(plat_marks=['cpu_windows'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_mindrecord_parallel_write_on_windows():
    """
    Feature: Test write mindrecord on windows
    Description: Testing write_raw_data(data, True) with exception
    Expectation: Success
    """
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    writer = FileWriter(file_name=file_name, shard_num=1, overwrite=True)
    schema_json = {"file_name": {"type": "string"}, "label": {"type": "int32"}, "data": {"type": "bytes"}}
    writer.add_schema(schema_json, "test_schema")
    indexes = ["file_name", "label"]
    writer.add_index(indexes)
    for i in range(10):
        data = [{"file_name": str(i) + ".jpg", "label": i,
                 "data": b"\x10c\xb3w\xa8\xee$o&<q\x8c\x8e(\xa2\x90\x90\x96\xbc\xb1\x1e\xd4QER\x13?\xff"}]
        with pytest.raises(RuntimeError) as err:
            writer.write_raw_data(data, True)
        assert "Parallel writing is not supported on the Windows platform. " in str(err)

    if os.path.exists(file_name):
        os.remove(file_name)
    if os.path.exists(file_name + ".db"):
        os.remove(file_name + ".db")


if __name__ == "__main__":
    test_mindrecord_parallel_write_on_windows()
