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
""" tests_custom_pyboost_cpu """

import tempfile
import os
import mindspore as ms
from mindspore.ops import CustomOpBuilder
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_custom_cpu_string():
    """
    Feature: CustomOpBuilder auto-generation verification.
    Description: Trigger the CustomOpBuilder workflow when *source*, *op_def*, and *op_doc* are supplied as
                 **single strings**, and verify that the three auto-generated files are produced correctly.
    Expectation: Directory <build_dir>/custom_cpu/custom_cpu_auto_generate exists and contains
                 gen_custom_ops_def.cc, gen_ops_def.py, gen_ops_prim.py.
    """
    ms.set_device("CPU")
    ms.set_context(mode=ms.GRAPH_MODE, save_graphs=False, save_graphs_path="./graphs")
    with tempfile.TemporaryDirectory() as tmpdirname:
        CustomOpBuilder("custom_cpu",
                        "kernel_impl/add_cpu.cpp",
                        backend="CPU", op_def="ops_yaml/add_cpu.yaml",
                        op_doc="ops_yaml/add_cpu.yaml",
                        build_dir=tmpdirname).build()
        auto_generate_path = os.path.join(tmpdirname, "custom_cpu_auto_generate")
        assert os.path.isdir(auto_generate_path), \
            f"auto_generate_path {auto_generate_path} does not exist."

        expected_files = [
            "gen_custom_ops_def.cc",
            "gen_ops_def.py",
            "gen_ops_prim.py"
        ]
        for fname in expected_files:
            fpath = os.path.join(auto_generate_path, fname)
            assert os.path.isfile(fpath), \
                f"Expected auto-generated file {fpath} is missing."
            assert os.path.getsize(fpath) > 0, f"{fname} is empty."


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_custom_cpu_list():
    """
    Feature: CustomOpBuilder auto-generation verification.
    Description: Trigger the CustomOpBuilder workflow when *source*, *op_def*, and *op_doc* are supplied as
                 **lists of strings**, and verify that the three auto-generated files are produced correctly.
    Expectation: Directory <build_dir>/custom_cpu/custom_cpu_auto_generate exists and contains
                 gen_custom_ops_def.cc, gen_ops_def.py, gen_ops_prim.py.
    """

    ms.set_device("CPU")
    ms.set_context(mode=ms.GRAPH_MODE, save_graphs=False, save_graphs_path="./graphs")
    with tempfile.TemporaryDirectory() as tmpdirname:
        CustomOpBuilder("custom_cpu",
                        ["kernel_impl/add_cpu.cpp"],
                        backend="CPU", op_def=["ops_yaml/add_cpu.yaml"],
                        op_doc=["ops_yaml/add_cpu.yaml"],
                        build_dir=tmpdirname).build()
        auto_generate_path = os.path.join(tmpdirname, "custom_cpu_auto_generate")
        assert os.path.isdir(auto_generate_path), \
            f"auto_generate_path {auto_generate_path} does not exist."

        expected_files = [
            "gen_custom_ops_def.cc",
            "gen_ops_def.py",
            "gen_ops_prim.py"
        ]
        for fname in expected_files:
            fpath = os.path.join(auto_generate_path, fname)
            assert os.path.isfile(fpath), \
                f"Expected auto-generated file {fpath} is missing."
            assert os.path.getsize(fpath) > 0, f"{fname} is empty."


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_custom_cpu_tuple():
    """
    Feature: CustomOpBuilder auto-generation verification.
    Description: Trigger the CustomOpBuilder workflow when *source*, *op_def*, and *op_doc* are supplied as
                 **tuples of strings**, and verify that the three auto-generated files are produced correctly.
    Expectation: Directory <build_dir>/custom_cpu/custom_cpu_auto_generate exists and contains
                 gen_custom_ops_def.cc, gen_ops_def.py, gen_ops_prim.py.
    """

    ms.set_device("CPU")
    ms.set_context(mode=ms.GRAPH_MODE, save_graphs=False, save_graphs_path="./graphs")
    with tempfile.TemporaryDirectory() as tmpdirname:
        CustomOpBuilder("custom_cpu",
                        ("kernel_impl/add_cpu.cpp"),
                        backend="CPU", op_def=("ops_yaml/add_cpu.yaml"),
                        op_doc=("ops_yaml/add_cpu.yaml"),
                        build_dir=tmpdirname).build()
        auto_generate_path = os.path.join(tmpdirname, "custom_cpu_auto_generate")
        assert os.path.isdir(auto_generate_path), \
            f"auto_generate_path {auto_generate_path} does not exist."

        expected_files = [
            "gen_custom_ops_def.cc",
            "gen_ops_def.py",
            "gen_ops_prim.py"
        ]
        for fname in expected_files:
            fpath = os.path.join(auto_generate_path, fname)
            assert os.path.isfile(fpath), \
                f"Expected auto-generated file {fpath} is missing."
            assert os.path.getsize(fpath) > 0, f"{fname} is empty."


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_custom_cpu_mixed():
    """
    Feature: CustomOpBuilder auto-generation verification.
    Description: Trigger the CustomOpBuilder workflow when *source*, *op_def*, and *op_doc* are supplied as
                 **mixed types** (source as string, op_def as tuple, op_doc as string), and verify that the
                 three auto-generated files are produced correctly.
    Expectation: Directory <build_dir>/custom_cpu/custom_cpu_auto_generate exists and contains
                 gen_custom_ops_def.cc, gen_ops_def.py, gen_ops_prim.py.
    """

    ms.set_device("CPU")
    ms.set_context(mode=ms.GRAPH_MODE, save_graphs=False, save_graphs_path="./graphs")
    with tempfile.TemporaryDirectory() as tmpdirname:
        CustomOpBuilder("custom_cpu",
                        "kernel_impl/add_cpu.cpp",
                        backend="CPU", op_def=("ops_yaml/add_cpu.yaml"),
                        op_doc="ops_yaml/add_cpu.yaml",
                        build_dir=tmpdirname).build()
        auto_generate_path = os.path.join(tmpdirname, "custom_cpu_auto_generate")
        assert os.path.isdir(auto_generate_path), \
            f"auto_generate_path {auto_generate_path} does not exist."

        expected_files = [
            "gen_custom_ops_def.cc",
            "gen_ops_def.py",
            "gen_ops_prim.py"
        ]
        for fname in expected_files:
            fpath = os.path.join(auto_generate_path, fname)
            assert os.path.isfile(fpath), \
                f"Expected auto-generated file {fpath} is missing."
            assert os.path.getsize(fpath) > 0, f"{fname} is empty."
