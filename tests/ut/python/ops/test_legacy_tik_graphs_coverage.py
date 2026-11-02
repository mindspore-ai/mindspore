# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Build real legacy Tik IR; inspect its binary-emission request without using an NPU.

These tests do not execute Ascend arithmetic or assert numerical equivalence. The
real SDK constructs every Tensor, Scalar, loop, condition and instruction. Only
BuildCCE is replaced, after construction, to inspect the actual TVM statement and
its ordered GM ABI. The SDK's hardware profile state is restored after each test.
"""
import importlib

import pytest

pytest.importorskip('te', reason='Real Tik IR tests require the optional CANN TE SDK')
from te import tik, tvm
from tbe.common import platform


def legacy(name):
    """Import the installed source-backed module normally, without extracted code."""
    return importlib.import_module('mindspore.ops._op_impl._custom_op.' + name)


def descriptor(shape, dtype='float16'):
    return {'shape': shape, 'dtype': dtype}


@pytest.fixture(autouse=True)
def restore_sdk_profile():
    """Dprofile selects global SDK configuration even when no binary is emitted."""
    previous = tuple(platform.get_soc_spec(key) for key in ('SOC_VERSION', 'CORE_TYPE', 'CORE_NUM'))
    profile_globals = tik.Dprofile.__mro__[1].__init__.__globals__
    product_flag = profile_globals['SET_PRODUCT_FLAG']
    previous_flag = dict(product_flag)
    soc_info = profile_globals['TIK_SOC_INFO']
    previous_info = vars(soc_info).copy()
    source_info = profile_globals['TikSourceInfo']
    was_initialized = source_info.is_inited
    source_globals = source_info.__init__.__globals__
    debug_states = [(source_globals[name], vars(source_globals[name]).copy())
                    for name in ('TIK_CONTROL', 'TIK_ERROR_MSG', 'ERROR_MSG_LEVEL')]
    # Give each graph fresh SDK source-location metadata. In particular, an
    # unsupported Dprofile can throw before clearing its source frame.
    for state, _ in debug_states:
        vars(state).clear()
    source_info.is_inited = False
    try:
        # Cloud kernels explicitly request v100/cloud (Ascend910). Let the SDK
        # honor those arguments, even if another test selected a product before.
        product_flag['is_set_product_version'] = False
        platform.set_current_compile_soc_info('Ascend910')
        yield
    finally:
        # CANN 9.1 reports CORE_TYPE='unknown' even for an AiCore target;
        # its setter accepts AiCore and the observable spec remains 'unknown'.
        core_type = 'AiCore' if previous[1] == 'unknown' else previous[1]
        platform.set_current_compile_soc_info(previous[0], core_type=core_type, aicore_num=previous[2])
        assert tuple(platform.get_soc_spec(key) for key in ('SOC_VERSION', 'CORE_TYPE', 'CORE_NUM')) == previous
        product_flag.clear()
        product_flag.update(previous_flag)
        vars(soc_info).clear()
        vars(soc_info).update(previous_info)
        for state, previous_state in debug_states:
            vars(state).clear()
            vars(state).update(previous_state)
        source_info.is_inited = was_initialized


@pytest.fixture
def build_requests(monkeypatch, tmp_path):
    """Intercept exactly final binary emission and inspect the already built IR."""
    requests = []

    def capture(builder, *, kernel_name, inputs, outputs):
        statement = builder.get()
        assert isinstance(statement, tvm.tir.Stmt)
        calls, blocks, allocations = [], [], {}

        def visit(node):
            if isinstance(node, tvm.tir.Call) and node.op.name == 'tir.call_extern':
                calls.append(node)
            if isinstance(node, tvm.tir.AttrStmt) and node.attr_key == 'thread_extent':
                blocks.append(int(node.value))
            if isinstance(node, tvm.tir.Allocate):
                allocations[node.buffer_var.name] = (str(node.dtype), tuple(int(n) for n in node.extents))

        tvm.tir.stmt_functor.post_order_visit(statement, visit)
        assert calls, 'The captured program must contain actual accelerator instructions'
        (tmp_path / (kernel_name + '.tir')).write_text(str(statement))
        requests.append({'builder': builder, 'name': kernel_name, 'inputs': inputs, 'outputs': outputs,
                         'calls': calls, 'blocks': blocks, 'allocations': allocations})

    monkeypatch.setattr(tik.Tik, 'BuildCCE', capture)
    return requests


def assert_abi(request, inputs, outputs):
    """Check explicit external layout fixtures, including transposed gradient storage."""
    for actual, expected in ((request['inputs'], inputs), (request['outputs'], outputs)):
        assert [(tensor.name, tuple(tensor.shape), tensor.dtype) for tensor in actual] == expected
        assert all(tensor.scope == tik.scope_gm for tensor in actual)
    names = [call.args[0].value for call in request['calls']]
    assert {'copy_gm_to_cbuf', 'load_cbuf_to_ca', 'load_cbuf_to_cb', 'mad', 'copy_ubuf_to_gm'} <= set(names)
    writes = [call for call in request['calls'] if call.args[0].value == 'copy_ubuf_to_gm']
    written_buffers = {call.args[1].args[1].name for call in writes}
    assert written_buffers == {tensor[0] for tensor in outputs}
    for call in request['calls']:
        if call.args[0].value == 'mad':
            # Accumulate in L0C float32, loading both tiled float16 cube operands.
            assert str(call.args[1].dtype) == 'float32'
            assert str(call.args[2].dtype) == str(call.args[3].dtype) == 'float16'


@pytest.mark.parametrize('batch,heads', [(1, 4), (2, 8)])
def test_dds_attention_builds_real_masked_softmax_ir(build_requests, batch, heads):
    """1024 tokens, 128 dimensions/head, local 64 and global 256 use their documented layouts."""
    q_shape = (heads * 8, batch * 64, 16, 16)
    local = (batch, heads, 16, 4, 4, 16, 16)
    global_ = (batch, heads, 16, 16, 4, 16, 16)
    result = legacy('matmul_dds_impl').matmul_dds(
        descriptor(q_shape), descriptor(q_shape), descriptor((64, batch * 4, 16, 16), 'float32'),
        descriptor((batch * 16, 64, 16, 16), 'float32'), descriptor(local), descriptor(global_),
        batch, heads, kernel_name='coverage_dds')
    request, = build_requests
    assert result is request['builder']
    assert request['name'] == 'coverage_dds'
    assert request['blocks'] == [batch * heads]
    assert_abi(request,
               [('mat_q', q_shape, 'float16'), ('mat_k', q_shape, 'float16'),
                ('mat_lm', (64, batch * 4, 16, 16), 'float32'),
                ('mat_gm', (batch * 16, 64, 16, 16), 'float32')],
               [('mat_lc', local, 'float16'), ('mat_gc', global_, 'float16')])
    names = {call.args[0].value for call in request['calls']}
    assert {'vcmax', 'vexp', 'vrec', 'vmul', 'vconv_f322f16'} <= names
    assert request['allocations']['mat_l1_ones'] == ('float16', (5120,))


@pytest.mark.parametrize('batch,heads', [(1, 4), (2, 8)])
def test_dds_gradient_builds_real_softmax_and_transposed_key_gradient_ir(build_requests, batch, heads):
    """dq keeps q's zN layout; dk has the reversed leading axes required by the ABI."""
    q_shape = (heads * 8, batch * 64, 16, 16)
    local = (batch, heads, 16, 4, 4, 16, 16)
    global_ = (batch, heads, 16, 16, 4, 16, 16)
    result = legacy('matmul_dds_grad_impl').matmul_dds_grad(
        descriptor(q_shape), descriptor(q_shape), descriptor(local), descriptor(global_),
        descriptor(local), descriptor(global_), descriptor(q_shape),
        descriptor((batch * 64, heads * 8, 16, 16)), kernel_name='coverage_dds_gradient')
    request, = build_requests
    assert result is request['builder']
    assert request['name'] == 'coverage_dds_gradient'
    assert request['blocks'] == [batch * heads]
    assert_abi(request,
               [('mat_q', q_shape, 'float16'), ('mat_k', q_shape, 'float16'),
                ('mat_lc', local, 'float16'), ('mat_gc', global_, 'float16'),
                ('mat_lc_grad', local, 'float16'), ('mat_gc_grad', global_, 'float16')],
               [('mat_dq', q_shape, 'float16'), ('mat_dk', (batch * 64, heads * 8, 16, 16), 'float16')])
    names = {call.args[0].value for call in request['calls']}
    assert {'vmul', 'vsub', 'vadd', 'vtranspose'} <= names


@pytest.mark.parametrize('batch,heads,sequence,embedding', [(1, 4, 1024, 64), (2, 8, 1024, 128),
                                                         (1, 4, 2048, 64)])
@pytest.mark.parametrize('backward', [False, True])
def test_dsd_builds_real_local_and_global_matrix_paths(build_requests, batch, heads, sequence, embedding, backward):
    """Exercise both backward tiling choices and inspect the ordered forward/gradient ABI."""
    block_count, global_size = sequence // 64, sequence // 4
    w1 = (batch, heads, block_count, 4, 4, 16, 16)
    w2 = (batch, heads, block_count, global_size // 16, 4, 16, 16)
    v = (batch * sequence // 16, heads * embedding // 16, 16, 16)
    y = (batch, heads, embedding // 16, sequence // 16, 16, 16)
    arguments = [descriptor(w1), descriptor(w2), descriptor(v)]
    if backward:
        result = legacy('dsd_back_impl').dsdbpropimpl(*arguments, descriptor(y), descriptor(y),
                                                    kernel_name='coverage_dsd_gradient')
    else:
        result = legacy('dsd_impl').dsd_matmul(*arguments, descriptor(y), kernel_name='coverage_dsd')
    request, = build_requests
    assert result is request['builder']
    assert request['name'] == ('coverage_dsd_gradient' if backward else 'coverage_dsd')
    assert request['blocks'] == [batch * heads]
    inputs = [('w1_gm', w1, 'float16'), ('w2_gm', w2, 'float16'), ('v_gm', v, 'float16')]
    outputs = [('output_gm', y, 'float16')]
    if backward:
        inputs += [('a_gm', y, 'float16'), ('d_a_gm', y, 'float16')]
        outputs = [('d_w1_gm', w1, 'float16'), ('d_w2_gm', w2, 'float16'), ('d_v_gm', v, 'float16')]
    assert_abi(request, inputs, outputs)
    cubes = [call for call in request['calls'] if call.args[0].value == 'mad']
    dimensions = {tuple(int(arg) for arg in call.args[4:7]) for call in cubes}
    if backward:
        ub_splits = 1 if sequence == 1024 else 2
        k_splits = max(1, global_size * embedding * 4 // 65536)
        assert (64, 64, embedding) in dimensions
        assert (64, embedding, 64) in dimensions
        assert (64 // ub_splits, embedding // k_splits, global_size) in dimensions
        assert len(cubes) == 6
        assert request['allocations']['d_a_l1'] == ('float16', (sequence * embedding,))
        names = {call.args[0].value for call in request['calls']}
        assert {'vtranspose', 'vadd', 'vconv_f322f16'} <= names
    else:
        assert dimensions == {(64, 64, embedding), (64, global_size // (sequence // 512), embedding)}
        assert len(cubes) == 3
        assert request['allocations']['v_global_l1'] == ('float16', (global_size * embedding,))


@pytest.mark.xfail(strict=True, raises=RuntimeError,
                   reason='CANN 9.1 lacks the legacy Ascend310 platform configuration requested by v100/mini')
def test_legacy_mini_sdk_profile_compatibility():
    """Record the SDK limitation without substituting product-discovery metadata.

    The old DSD mini branches cannot be selected by this SDK's product report:
    cce_conf uses '1910' whereas topi.util.VERSION_MINI is 'mini'. Independently
    requesting the exact real Dprofile used by those branches also fails. This
    compatibility xfail does not count as successful kernel line/IR coverage.
    """
    try:
        tik.Dprofile('v100', 'mini')
    except RuntimeError as error:
        assert 'get platform info failed' in str(error), str(error)
        raise
