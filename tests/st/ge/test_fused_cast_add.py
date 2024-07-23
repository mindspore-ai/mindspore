# Copyright 2024 Huawei Technologies Co., Ltd
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
import os
import pytest
from tests.mark_utils import arg_mark
from mindspore import context
from mindspore import Tensor, nn
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
from mindspore.common import JitConfig

ir_dir = "./fused_cast_add"
before_opt_ge_prefix = "hwopt_d_before_opt_ge"
last_ms_backend_graph_prefix = "anf_graph_after"
cast_operator_keyword = "Cast("

context.set_context(mode=context.GRAPH_MODE,
                    device_target="Ascend",
                    save_graphs=True,
                    save_graphs_path=ir_dir)


def check_file_exists(startswith, path=ir_dir):
    for file in os.listdir(path):
        if file.startswith(startswith):
            return os.path.join(path, file)
    return None


def search_keyword_in_file(keyword, ir_file):
    with open(ir_file, 'r') as f:
        for line in f:
            if keyword in line:
                return True
    return False


def delete_previous_graph(dir_path):
    if os.path.isdir(dir_path):
        os.system(f"rm -r {dir_path}")


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('compute_dtype', [mstype.float16, mstype.bfloat16])
@pytest.mark.parametrize('enable_fused_cast_add_opt', [True, False])
def test_fused_cast_add(compute_dtype, enable_fused_cast_add_opt):
    """
    Description: Test operator <cast>, <add> are fused under ge.
    Expectation: Run without errors.
    """
    delete_previous_graph(ir_dir)
    context.set_context(enable_fused_cast_add_opt=enable_fused_cast_add_opt)
    class Net16(nn.Cell):
        def __init__(self):
            super().__init__()
            self.cast = P.Cast()
            self.add = P.Add()

        def construct(self, x1, x2, y):
            out1 = self.add(x1, x2)
            out2 = self.cast(out1, mstype.float32)
            out = self.add(y, out2)
            return out

    net = Net16()
    net.set_jit_config(JitConfig(jit_level="O2"))
    x1 = Tensor(2000., dtype=compute_dtype)
    x2 = Tensor(2000., dtype=compute_dtype)
    y = Tensor(1000., dtype=mstype.float32)
    net(x1, x2, y)
    before_ge_opt_ir = check_file_exists(before_opt_ge_prefix)
    last_backend_ir = check_file_exists(last_ms_backend_graph_prefix)
    if not before_ge_opt_ir or not last_backend_ir:
        raise Exception(f"""Could not find the necessary intermediate graphs!\n \
                        Please check files with prefixs \
                            {before_opt_ge_prefix} and {last_ms_backend_graph_prefix} exist""")
    found_cast_before_opt = search_keyword_in_file(
        cast_operator_keyword, before_ge_opt_ir)
    if not found_cast_before_opt:
        raise ValueError(
            "Cast are not found before the optimization. Please check if setup are properly set.")
    found_cast_after_opt = search_keyword_in_file(
        cast_operator_keyword, last_backend_ir)
    if found_cast_after_opt and enable_fused_cast_add_opt:
        raise ValueError(
            "Cast should have been fused after the optimization but still exists.")
