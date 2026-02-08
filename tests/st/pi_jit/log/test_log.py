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
"""test pijit log"""
import os
import subprocess
from tests.mark_utils import arg_mark
import re


def run_test_case_common(test_file_name, test_case_name, log_config):
    cmd = f"export MS_JIT_BYTECODE_LOGS={log_config}; export GLOG_v=2; export GLOG_logtostderr=1;" \
          + f"python {test_file_name} > {test_case_name}_log.txt 2>&1"
    subprocess.check_output(cmd, shell=True)
    with open(f"{test_case_name}_log.txt", "r") as f:
        data = f.read()
    return data


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_pijit_log_guard():
    """
    Feature: Test pijit log
    Description: Test different pijit log config.
    Expectation: expect to get correct log info.
    """
    test_case_name = "test_pijit_log_guard"
    log_config = "guard"
    data = run_test_case_common("basic_function.py", test_case_name, log_config)

    assert "[guard]" in data

    os.remove(f"{test_case_name}_log.txt")


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_pijit_log_bytecode():
    """
    Feature: Test pijit log
    Description: Test different pijit log config.
    Expectation: expect to get correct log info.
    """
    test_case_name = "test_pijit_log_bytecode"
    log_config = "bytecode"
    data = run_test_case_common("basic_function.py", test_case_name, log_config)

    assert "ORIGINAL BYTECODE of" in data

    os.remove(f"{test_case_name}_log.txt")


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_pijit_log_graph_break():
    """
    Feature: Test pijit log
    Description: Test different pijit log config.
    Expectation: expect to get correct log info.
    """
    test_case_name = "test_pijit_log_graph_break"
    log_config = "graph_break"
    data = run_test_case_common("basic_function.py", test_case_name, log_config)

    assert "[graph_break]" in data

    os.remove(f"{test_case_name}_log.txt")


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_pijit_log_all():
    """
    Feature: Test pijit log
    Description: Test different pijit log config.
    Expectation: expect to get correct log info.
    """
    test_case_name = "test_pijit_log_all"
    log_config = "all"
    data = run_test_case_common("basic_function.py", test_case_name, log_config)

    assert "[guard]" in data
    assert "ORIGINAL BYTECODE of" in data
    assert "[graph_break]" in data

    os.remove(f"{test_case_name}_log.txt")


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_pijit_log_mix_guard_bytecode():
    """
    Feature: Test pijit log
    Description: Test different pijit log config.
    Expectation: expect to get correct log info.
    """
    test_case_name = "test_pijit_log_mix_guard_bytecode"
    log_config = "guard,bytecode"
    data = run_test_case_common("basic_function.py", test_case_name, log_config)

    assert "[guard]" in data
    assert "ORIGINAL BYTECODE of" in data
    assert "[graph_break]" not in data

    os.remove(f"{test_case_name}_log.txt")


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_pijit_log_mix_guard_bytecode_2():
    """
    Feature: Test pijit log
    Description: Test different pijit log config.
    Expectation: expect to get correct log info.
    """
    test_case_name = "test_pijit_log_mix_guard_bytecode_2"
    log_config = "\" guard , bytecode \""
    data = run_test_case_common("basic_function.py", test_case_name, log_config)

    assert "[guard]" in data
    assert "ORIGINAL BYTECODE of" in data
    assert "[graph_break]" not in data

    os.remove(f"{test_case_name}_log.txt")


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_pijit_log_mix_guard_all():
    """
    Feature: Test pijit log
    Description: Test different pijit log config.
    Expectation: expect to get correct log info.
    """
    test_case_name = "test_pijit_log_mix_guard_all"
    log_config = "guard,all"
    data = run_test_case_common("basic_function.py", test_case_name, log_config)

    assert "[guard]" in data
    assert "ORIGINAL BYTECODE of" in data
    assert "[graph_break]" in data

    os.remove(f"{test_case_name}_log.txt")


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_pijit_log_graph_break_content():
    """
    Feature: Test pijit log
    Description: Test pijit log content.
    Expectation: expect correct output.
    """
    test_case_name = "test_pijit_log_graph_break_content"
    log_config = "graph_break"
    data = run_test_case_common("basic_function.py", test_case_name, log_config)

    assert "Got an unexpected keyword argument 'flush'" in data
    assert "print(\"Hi\", flush=True)" in data
    assert re.search(r"Graph break at.*basic_function\.py:23", data)

    os.remove(f"{test_case_name}_log.txt")


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_pijit_log_trace_content():
    """
    Feature: Test pijit log
    Description: Test pijit log content.
    Expectation: expect correct output.
    """
    test_case_name = "test_pijit_log_trace_content"
    log_config = "trace_source,trace_bytecode"
    data = run_test_case_common("basic_function.py", test_case_name, log_config)

    assert "z = ops.add(x, y)" in data
    assert "10 CALL_FUNCTION 2" in data
    assert "[Function:add, Tensor:{shape=(4,), type=Float32}, Tensor:{shape=(4,), type=Float32}]" in data

    os.remove(f"{test_case_name}_log.txt")


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_pijit_log_recompiles_content():
    """
    Feature: Test pijit log
    Description: Test pijit log content.
    Expectation: expect correct output.
    """
    test_file_name = "recompile_function.py"
    test_case_name = "test_pijit_log_recompiles_content"
    log_config = "recompiles_verbose"
    data = run_test_case_common(test_file_name, test_case_name, log_config)

    assert re.search(r"Recompile func.*recompile_function\.py.*19", data)

    os.remove(f"{test_case_name}_log.txt")


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_pijit_log_dynamic_content():
    """
    Feature: Test pijit log
    Description: Test pijit log content.
    Expectation: expect correct output.
    """
    test_file_name = "dynamic_function.py"
    test_case_name = "test_pijit_log_dynamic_content"
    log_config = "dynamic"
    data = run_test_case_common(test_file_name, test_case_name, log_config)

    assert re.search(r"dynamic.*self\.x at.*dynamic_function\.py.*27", data)
    assert "Symbolic object value: Tensor" in data

    os.remove(f"{test_case_name}_log.txt")


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_pijit_log_bytecode_graph_split_case():
    """
    Feature: Test pijit log (bytecode only).
    Description: Enable bytecode log and run a graph-split helper script.
    Expectation: Expect to see bytecode log content.
    Migrated from: test_parse_pijit_improve_debug_ability.py::test_parse_pijit_improve_debug_ability_001
    """
    test_case_name = "test_pijit_log_bytecode_graph_split_case"
    log_config = "bytecode"
    data = run_test_case_common("graph_split_case.py", test_case_name, log_config)

    assert "1 passed" in data
    assert "ORIGINAL BYTECODE of" in data

    os.remove(f"{test_case_name}_log.txt")


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_pijit_log_graph_break_and_guard_graph_split_case():
    """
    Feature: Test pijit log (graph_break and guard).
    Description: Enable graph_break and guard logs and run a graph-split helper script.
    Expectation: Expect to see graph_break and guard logs, but no bytecode log.
    Migrated from: test_parse_pijit_improve_debug_ability.py::test_parse_pijit_improve_debug_ability_002
    """
    test_case_name = "test_pijit_log_graph_break_and_guard_graph_split_case"
    log_config = "graph_break,guard"
    data = run_test_case_common("graph_split_case.py", test_case_name, log_config)

    assert "1 passed" in data
    assert "generated guard at" in data  # guard
    assert "[guard]" in data
    assert "[graph_break]" in data
    assert "ORIGINAL BYTECODE of" not in data

    os.remove(f"{test_case_name}_log.txt")


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_pijit_log_all_graph_split_case():
    """
    Feature: Test pijit log (all).
    Description: Enable all logs and run a graph-split helper script.
    Expectation: Expect to see guard, bytecode and graph_break logs.
    Migrated from: test_parse_pijit_improve_debug_ability.py::test_parse_pijit_improve_debug_ability_003
    """
    test_case_name = "test_pijit_log_all_graph_split_case"
    log_config = "all"
    data = run_test_case_common("graph_split_case.py", test_case_name, log_config)

    assert "1 passed" in data
    assert "generated guard at" in data  # guard
    assert "UD analyze: enter GetAliveNodes" in data  # others
    assert "[guard]" in data
    assert "ORIGINAL BYTECODE of" in data
    assert "[graph_break]" in data

    os.remove(f"{test_case_name}_log.txt")


def run_pytest(test_file_name, test_case_name, log_config):
    cmd = f"export MS_JIT_BYTECODE_LOGS={log_config}; export GLOG_v=2; export GLOG_logtostderr=1;" \
          + f"python -m pytest -vs {test_file_name}::{test_case_name} > {test_case_name}_log.txt 2>&1"
    subprocess.check_output(cmd, shell=True)
    with open(f"{test_case_name}_log.txt", "r") as f:
        data = f.read()
    return data


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_pijit_log_print_after_all_graph_split_for():
    """
    Feature: print_after_all logging for graph break in loop.
    Description: Enable print_after_all and execute the graph split helper to trigger traceback bytecode dump.
    Expectation: Log output contains traceback bytecode dump information.
    Migrated from: test_pijit_print_after_all.py::test_pijit_print_after_all_graph_split_for
    """
    test_case_name = "test_print_after_all_graph_split_for"
    data = run_pytest("print_after_all_case.py", test_case_name, "")

    assert "*** Dump ByteCode After Traceback on [construct] ***" in data
    assert "1 passed" in data

    os.remove(f"{test_case_name}_log.txt")


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_pijit_log_print_after_all_try_finally():
    """
    Feature: print_after_all logging for try/finally graph break.
    Description: Enable print_after_all and execute the try/finally helper to dump codegen bytecode.
    Expectation: Log output contains one-stage bytecode collection and final bytecode dump.
    Migrated from: test_pijit_print_after_all.py::test_pijit_print_after_all_try_finally_break
    """
    test_case_name = "test_print_after_all_try_finally_break"
    data = run_pytest("print_after_all_case.py", test_case_name, "")

    assert "*** Dump One Stage ByteCode Collection After CodeGen ***" in data
    assert "MODIFIED BYTECODE of" in data
    assert "1 passed" in data

    os.remove(f"{test_case_name}_log.txt")


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_pijit_log_print_after_all_before_psjit():
    """
    Feature: print_after_all logging before psjit execution.
    Description: Enable print_after_all with a mixed bytecode/ast pipeline to verify codegen bytecode dumps.
    Expectation: Log output contains one-stage bytecode collection and final bytecode dump.
    Migrated from: test_pijit_print_after_all.py::test_pijit_print_after_all_before_psjit
    """
    test_case_name = "test_print_after_all_before_psjit"
    data = run_pytest("print_after_all_case.py", test_case_name, "")

    assert "*** Dump One Stage ByteCode Collection After CodeGen ***" in data
    assert "MODIFIED BYTECODE of" in data
    assert "1 passed" in data

    os.remove(f"{test_case_name}_log.txt")
