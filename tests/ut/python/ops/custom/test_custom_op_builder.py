# Copyright 2025 Huawei Technologies Co., Ltd
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
""" tests_custom_op_builder """
import os
import tempfile
import pytest
from mindspore.ops import CustomOpBuilder


@pytest.fixture(scope="function", autouse=True)
def isolate_env_and_clean(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ASCEND_OPP_PATH", "/usr/local/Ascend/opp")
    monkeypatch.setenv("ATB_HOME_PATH", "/usr/local/Ascend/atb")
    yield


@pytest.mark.parametrize(
    "kwargs",
    [

        {"name": "cpu0", "sources": "a.cc", "backend": "CPU"},
        {"name": "cpu1", "sources": ["a.cc", "b.cc"], "backend": "CPU"},
        {"name": "cpu2", "sources": ("a.cc",), "backend": "CPU"},
        {"name": "cpu3", "sources": "a.cc", "include_paths": "inc"},
        {"name": "cpu4", "sources": "a.cc", "include_paths": ["inc1", "inc2"]},
        {"name": "cpu5", "sources": "a.cc", "cflags": "-Wall"},
        {"name": "cpu6", "sources": "a.cc", "ldflags": "-lfoo"},
        {"name": "cpu7", "sources": "a.cc", "debug_mode": True},
        {"name": "cpu8", "sources": "a.cc", "build_dir": "/tmp/my_build"},
        {"name": "asc0", "sources": "a.cc", "backend": "Ascend"},
        {"name": "asc1", "sources": "a.cc", "backend": "Ascend", "enable_atb": True},
        {"name": "asc2", "sources": "a.cc", "backend": "Ascend", "enable_asdsip": True},
        {"name": "asc3", "sources": "a.cc", "backend": "Ascend",
         "enable_atb": True, "enable_asdsip": False},
        {"name": "asc4", "sources": "a.cc", "enable_atb": True},
        {"name": "mix0",
         "sources": ["a.cc", "b.cc"],
         "backend": "Ascend",
         "include_paths": ["inc1", "inc2"],
         "cflags": "-O3 -Wall",
         "ldflags": "-lfoo -lbar",
         "build_dir": "/tmp/mix0_build"},
    ]
)
def test_custom_op_builder_valid_args(kwargs):
    """
    Feature: test custom op parameter validation
    Description: pass legal arguments
    Expectation: success
    """
    builder = CustomOpBuilder(**kwargs)
    assert isinstance(builder.name, str)
    assert builder.name == kwargs["name"]


@pytest.mark.parametrize("kwargs,expect_type,expect_msg", [
    ({"name": 123, "sources": "op.cc"}, TypeError, "name"),
    ({"name": "Op", "sources": 123}, TypeError, "sources"),
    ({"name": "Op", "sources": ["a.cc", 123]}, TypeError, "sources"),
    ({"name": "Op", "sources": "a.cc", "backend": "GPU"}, ValueError, "backend"),
    ({"name": "Op", "sources": "a.cc", "include_paths": ["inc", 123]}, TypeError, "include_paths"),
    ({"name": "Op", "sources": "a.cc", "cflags": 123}, TypeError, "cflags"),
    ({"name": "Op", "sources": "a.cc", "ldflags": 123}, TypeError, "ldflags"),
    ({"name": "Op", "sources": "a.cc", "debug_mode": "true"}, TypeError, "debug_mode"),
    ({"name": "Op", "sources": "a.cc", "op_def": 123}, TypeError, "op_def"),
    ({"name": "Op", "sources": "a.cc", "op_def": ["x.yaml", 123]}, TypeError, "op_def"),
    ({"name": "Op", "sources": "a.cc", "op_doc": 123}, TypeError, "op_doc"),
    ({"name": "Op", "sources": "a.cc", "backend": "CPU", "enable_atb": True}, ValueError, "enable_atb"),
])
def test_custom_op_builder_invalid_args(kwargs, expect_type, expect_msg):
    """
    Feature: test custom op parameter validation
    Description: pass illegal arguments and expect exceptions
    Expectation: raises the specified exception type with matching message
    """
    with pytest.raises(expect_type, match=f".*{expect_msg}.*"):
        CustomOpBuilder(**kwargs)


def test_custom_op_builder_gen_op_def():
    """
    Feature: test CustomOpBuilder generate function
    Description: generate files by yaml
    Expectation: success
    """

    def is_file_nonempty(path: str) -> bool:
        return os.path.isfile(path) and os.path.getsize(path) > 0

    with tempfile.TemporaryDirectory() as tmpdirname:
        script_path, _ = os.path.split(__file__)
        builder = CustomOpBuilder("op",
                                  "a.cc",
                                  backend="Ascend", op_def=[os.path.join(script_path, "ops_yaml/inplace_add.yaml")],
                                  build_dir=tmpdirname)
        # pylint: disable=protected-access
        builder._get_op_def()
        assert is_file_nonempty(os.path.join(tmpdirname, "op_auto_generate", "gen_custom_ops_def.cc"))
        assert is_file_nonempty(os.path.join(tmpdirname, "op_auto_generate", "gen_ops_def.py"))
        assert is_file_nonempty(os.path.join(tmpdirname, "op_auto_generate", "gen_ops_prim.py"))
