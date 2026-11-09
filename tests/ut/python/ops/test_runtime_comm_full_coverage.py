# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Communication protocol unit tests and real CPU local Tensor computations.

External communication endpoints return explicitly supplied peer responses. These
are protocol tests, not end-to-end distributed communication correctness tests.
"""
# Unit coverage intentionally exercises internal Python inference and wrapper helpers.
# pylint: disable=protected-access
import builtins
import io
import inspect
import json
import logging
import os
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor, Parameter, context as C, log as L
from mindspore.common.sparse_tensor import RowTensorInner
from mindspore.ops.operations import comm_ops as O
from mindspore.ops.function import comm_func as F
from mindspore.ops._grad_experimental import grad_comm_ops as G


@pytest.fixture(autouse=True, scope='module', name='cpu_context')
def fixture_cpu_context():
    old = {k: C.get_context(k) for k in ('mode', 'device_target', 'pynative_synchronize')}
    C.set_context(mode=C.PYNATIVE_MODE, device_target='CPU', pynative_synchronize=True)
    yield
    C.set_context(**old)


def t(value, dtype=ms.float32):
    return Tensor(value, dtype)


def check(actual, expected):
    assert isinstance(actual, Tensor)
    np.testing.assert_allclose(actual.asnumpy(), expected, rtol=1e-6, atol=1e-6)


class HandleBoundary:
    """Isolated replacement for the C++ context storage, with exact API methods."""
    def __init__(self):
        self.params = {}
        self.dump = True

    def set_param(self, key, value):
        self.params[key] = value

    def get_param(self, key):
        return self.params.get(key, 'CPU')

    def set_backend_policy(self, policy):
        return policy in ('ge', 'vm', 'ms')

    def enable_dump_ir(self):
        return self.dump


@pytest.fixture(name='isolated_context')
def fixture_isolated_context():
    # Bypass singleton construction; never change the application's C++ context.
    ctx = object.__new__(C._Context)
    ctx._context_handle = HandleBoundary()
    ctx._thread_local_info = C._ThreadLocalInfo()
    ctx._context_switches = C._ContextSwitchInfo(False)
    ctx._support_binary = False
    ctx._mode = C.PYNATIVE_MODE
    ctx.gpu_config = {}
    ctx.jit_config = {}
    ctx.ascend_config = {}
    ctx.aoe_config = {}
    return ctx


@pytest.mark.parametrize('method,value,error', [
    ('set_mode', 9, ValueError), ('set_jit_syntax_level', 9, ValueError),
    ('set_debug_level', 9, ValueError), ('set_memory_optimize_level', 'bad', ValueError),
    ('set_exec_order', 'bad', ValueError), ('set_backend_policy', 'bad', RuntimeError),
    ('set_device_target', 'bad', ValueError), ('set_aoe_tune_mode', 'bad', ValueError),
    ('set_device_id', -1, ValueError), ('set_device_id', 4096, ValueError),
    ('set_max_call_depth', 0, ValueError), ('set_profiling_options', 1, TypeError),
    ('set_variable_memory_max_size', 'oops', ValueError),
    ('set_variable_memory_max_size', '32GB', ValueError),
    ('set_max_device_memory', 'oops', ValueError), ('set_max_device_memory', '0.0GB', ValueError),
    ('set_runtime_num_threads', -1, ValueError), ('set_op_timeout', -1, ValueError),
    ('set_inter_op_parallel_num', -1, ValueError),
    ('_set_topo_order', [], TypeError), ('_set_topo_order', {1:'dfs'}, TypeError),
    ('_set_topo_order', {'x':'bad'}, ValueError), ('_set_hccl_watchdog', 1, TypeError),
    ('_set_need_ckpt', 1, TypeError), ('_set_cur_step_num', '1', TypeError),
    ('_set_save_checkpoint_steps', '1', TypeError), ('_set_last_triggered_step', '1', TypeError),
    ('_set_op_debug_option', 'bad', ValueError),
    ('set_gpu_config', {'bad':'normal'}, ValueError),
    ('set_gpu_config', {'conv_fprop_algo':'bad'}, ValueError),
    ('set_aoe_config', {'bad':'1'}, ValueError), ('set_aoe_config', {'job_type':'bad'}, ValueError),
    ('set_jit_config', {'bad':'O0'}, ValueError),
    ('set_jit_config', {'jit_level':'O1', 'infer_boost':'on'}, ValueError),
])
def test_context_validation(isolated_context, method, value, error):
    """
    Feature: Context validation.
    Description: Check context validation.
    Expectation: Invalid context arguments raise the parameterized exception type.
    """
    with pytest.raises(error):
        getattr(isolated_context, method)(value)


def test_context_valid_storage_and_properties(isolated_context, monkeypatch):
    """
    Feature: Context valid storage and properties.
    Description: Check context valid storage and properties.
    Expectation: Context stores the explicit GPU/Ascend/topology options and property values and rejects invalid
        configuration fields.
    """
    ctx = isolated_context
    cfg = {'conv_fprop_algo': 'direct', 'conv_dgrad_algo': 'algo_0',
        'conv_wgrad_algo': 'algo_1', 'conv_allow_tf32': True, 'matmul_allow_tf32': False}
    ctx.set_gpu_config(cfg)
    assert ctx.gpu_config == cfg
    for key, value in cfg.items():
        assert ctx.get_param(getattr(C.ms_ctx_param, key)) == value
    for method, key, value in [
        ('set_debug_level', 'debug_level', C.DEBUG), ('set_aoe_tune_mode', 'aoe_tune_mode', 'online'),
        ('set_profiling_options', 'profiling_options', 'trace'),
        ('set_inter_op_parallel_num', 'inter_op_parallel_num', 2),
        ('_set_hccl_watchdog', 'hccl_watchdog', True), ('_set_need_ckpt', 'need_ckpt', True),
        ('_set_cur_step_num', 'cur_step_num', 3), ('_set_save_checkpoint_steps', 'save_checkpoint_steps', 4),
        ('_set_last_triggered_step', 'last_triggered_step', 5),
    ]:
        getattr(ctx, method)(value)
        assert ctx.get_param(getattr(C.ms_ctx_param, key)) == value
    ctx.set_aoe_config({'job_type':'1'})
    assert ctx.aoe_config == {'job_type':'1'}
    assert ctx.get_param(C.ms_ctx_param.aoe_job_type) == '1'
    ctx._set_topo_order({'op':'rdfs'})
    assert json.loads(ctx.get_param(C.ms_ctx_param.topo_order)) == {'op':'rdfs'}
    ctx.reserve_class_name_in_scope = False
    assert ctx.reserve_class_name_in_scope is False
    with pytest.raises(ValueError):
        ctx.reserve_class_name_in_scope = 1
    ctx.support_binary = True
    assert ctx.support_binary is True
    with pytest.raises(TypeError):
        ctx.support_binary = 1
    monkeypatch.setattr(C, '_need_reset_device_target_for_ps', lambda target: False)
    ctx.set_device_target('Davinci')
    assert ctx.get_param(C.ms_ctx_param.device_target) == 'Ascend'
    ctx._context_handle = None
    with pytest.raises(ValueError, match='env_config_path'):
        _ = ctx._context_handle


def test_context_files_and_switch_stack(monkeypatch, tmp_path):
    """
    Feature: Context files and switch stack.
    Description: Check context files and switch stack.
    Expectation: Valid directories and switch-stack push/pop preserve their expected paths and depths; invalid
        paths and permissions are rejected.
    """
    with pytest.raises(ValueError):
        C._make_directory(' ')
    assert C._make_directory(str(tmp_path)) == str(tmp_path)
    def denied(path, mode):
        raise PermissionError('injected')
    with monkeypatch.context() as patch:
        patch.setattr(C.os, 'makedirs', denied)
        with pytest.raises(ValueError, match='No write permission'):
            C._make_directory(str(tmp_path/'denied'))
    monkeypatch.setattr(C.time, 'time', lambda: 123)
    (tmp_path/'print.123').write_text('existing')
    with pytest.raises(ValueError, match='already exists'):
        C._get_print_file_name(str(tmp_path/'print'))
    stack = C._ContextSwitchInfo(True)
    calls = []
    stack.push(False, lambda: calls.append('switch'))
    assert calls == ['switch'] and len(stack.context_stack) == 2
    stack.pop()
    assert len(stack.context_stack) == 1


def test_context_json_and_memory(isolated_context, monkeypatch, tmp_path):
    """
    Feature: Context json and memory.
    Description: Check context json and memory.
    Expectation: Memory settings and valid JSON/config paths retain expected values, while missing files,
        invalid JSON, and unsupported options are rejected.
    """
    ctx = isolated_context
    monkeypatch.setattr(C, 'get_jit_config', lambda: {'backend':'GE'})
    ctx.set_mempool_block_size('invalid-but-ignored-by-ge')
    assert C.ms_ctx_param.mempool_block_size not in ctx._context_handle.params
    monkeypatch.setattr(C, 'get_jit_config', lambda: {'jit_level':'O0'})
    for bad in ('bad', '0.5GB'):
        with pytest.raises(ValueError):
            ctx.set_mempool_block_size(bad)
    ctx.set_mempool_block_size('2GB')
    assert ctx.get_param(C.ms_ctx_param.mempool_block_size) == 2.0
    ctx._context_handle.dump = False
    with pytest.raises(ValueError, match='not supported'):
        ctx.set_env_config_path(str(tmp_path/'absent'))
    ctx._context_handle.dump = True
    with pytest.raises(ValueError, match='not exists'):
        ctx.set_env_config_path(str(tmp_path/'absent'))
    path = tmp_path/'config.json'
    path.write_text('{')
    with pytest.raises(ValueError, match='json file'):
        ctx.set_env_config_path(str(path))
    path.write_text('{}')
    ctx.set_env_config_path(str(path))
    assert ctx.get_param(C.ms_ctx_param.env_config_path) == str(path)
    with pytest.raises(ValueError, match='does not exist'):
        ctx._set_speedup_config_path(str(tmp_path/'absent'))
    for cfg in ({'unknown':True}, {'enable_task_opt':'bad'}, {'pp_1f1b_overlap':'bad'}):
        path.write_text(json.dumps(cfg))
        with pytest.raises(ValueError):
            ctx._set_speedup_config_path(str(path))
    path.write_text(json.dumps({'recompute_comm_overlap':True, 'pp_1f1b_overlap':'AllReduce',
                               'enable_task_opt':True}))
    ctx._set_speedup_config_path(str(path))
    assert ctx.get_param(C.ms_ctx_param.recompute_comm_overlap).endswith('AllReduce')
    assert ctx.get_param(C.ms_ctx_param.pp_1f1b_overlap) == 'AllReduce'
    assert ctx._check_speedup_config_str_value('pp_1f1b_overlap', False) == ''


def test_context_public_validation(monkeypatch, isolated_context):
    """
    Feature: Context public validation.
    Description: Check context public validation.
    Expectation: Public context cache flags and parameter-server forwarding retain the expected values and
        removed/invalid options raise ValueError.
    """
    monkeypatch.setattr(C, 'K_CONTEXT', isolated_context)
    for key in ('exception_dump', 'precompile_only', 'grad_for_scalar'):
        with pytest.raises(ValueError, match='removed'):
            C._check_context_deprecated(key)
    C._check_context_deprecated('debug_level')
    C._check_context_deprecated('max_call_depth')
    C.set_context(enable_sparse=True, auto_tune_mode='NO_TUNE', enable_auto_mixed_precision=True)
    for kwargs in ({'save_graphs':4}, {'jit_syntax_level':9}, {'debug_level':9}, {'unrecognized':True}):
        with pytest.raises(ValueError):
            C.set_context(**kwargs)
    C.set_context(enable_compile_cache=True, enable_graph_kernel=False, graph_kernel_flags='')
    assert isolated_context.enable_compile_cache is True
    with pytest.raises(ValueError):
        C.get_context('unrecognized')
    monkeypatch.setattr(C, '_get_ps_context', lambda key: ('ps', key))
    assert C.get_ps_context('enable_ps') == ('ps', 'enable_ps')


@pytest.fixture(name='comm_environment')
def fixture_comm_environment(monkeypatch):
    monkeypatch.setattr(O, 'get_rank', lambda group: 1)
    monkeypatch.setattr(O, 'get_group_size', lambda group: 2)


def test_comm_primitive_inference(comm_environment):
    """
    Feature: Comm primitive inference.
    Description: Check comm primitive inference.
    Expectation: Communication primitives infer the explicit static/dynamic shapes and dtypes, preserve endpoint
        metadata, and reject invalid ranks or arguments.
    """
    # Table cases exercise heterogeneous decorated/native API signatures.
    # pylint: disable=too-many-function-args,no-value-for-parameter
    for ctor in (O.AllGather, O._MiniStepAllGather):
        prim = ctor('test-group')
        extra = ([0],) if ctor is O._MiniStepAllGather else ()
        assert prim.infer_shape([3,4], *extra) == [6,4]
        assert prim.infer_shape([-1,4], *extra) == [-1,4]
        dtype_extra = (ms.TensorType(ms.float32),) if extra else ()
        assert prim.infer_dtype(ms.TensorType(ms.float32), *dtype_extra) == ms.TensorType(ms.float32)
        with pytest.raises(ValueError):
            prim.infer_shape([], *extra)
    for ctor in (O._HostAllGather, O._HostReduceScatter):
        for group in (None, [0], [0,8]):
            with pytest.raises((ValueError, TypeError)):
                ctor(group=group)
        prim = ctor(group=[0,1])
        assert prim.infer_dtype(ms.TensorType(ms.float32)) == ms.TensorType(ms.float32)
        assert prim.infer_shape([4,3]) == ([8,3] if ctor is O._HostAllGather else [2,3])
        with pytest.raises(NotImplementedError):
            prim(t([1,2]))
    host = O._HostAllGather([0,1])
    assert host.infer_shape([-1,2]) == [-1,2]
    with pytest.raises(ValueError):
        O._HostReduceScatter(group=[0,1]).infer_shape([3,2])
    with pytest.raises(TypeError):
        O.AllReduce(op=1)
    with pytest.raises(TypeError):
        O.AllReduce(group=1)
    swap = O._AllSwap('test-group')
    out = swap.__check__({'shape':[4,3], 'dtype':ms.TensorType(ms.float32)},
                         {'shape':[2], 'dtype':ms.TensorType(ms.int64)},
                         {'shape':[2], 'dtype':ms.TensorType(ms.int64)})
    assert out['shape'] == [-1,3] and out['dtype'] == ms.TensorType(ms.float32)
    for ctor in (O._MirrorMiniStepOperator, O._VirtualAdd, O._VirtualAssignKvCache, O._VirtualOutput):
        prim = ctor()
        assert (prim.infer_shape([2,3]) if ctor is O._VirtualOutput else prim.infer_shape([2,3],[2,3])) == [2,3]
        assert (prim.infer_dtype(ms.float32) if ctor is O._VirtualOutput else prim.infer_dtype(
            ms.float32,ms.float32)) == ms.float32
    send = O.Send(0, 1, 'test-group')
    assert send.infer_shape([2,3]) == [2,3]
    assert send.infer_dtype(ms.float32) == ms.float32
    recv = O.Receive(0, 1, [2,3], ms.float32, 'test-group')
    assert recv.infer_shape() == [2,3] and recv.infer_dtype() == ms.float32
    batch = O.BatchISendIRecv(['isend'], [1], group=None)
    assert batch.receive_shapes == () and batch.receive_dtypes == ()
    with pytest.raises(ValueError):
        O.BatchISendIRecv([], [], [(2,)], [])
    with pytest.raises(ValueError):
        O.BatchISendIRecv(['isend'], [])


def test_comm_serialization_and_validation(monkeypatch):
    """
    Feature: Comm serialization and validation.
    Description: Check comm serialization and validation.
    Expectation: Objects round-trip through tensor encoding, padding is respected, restricted types and
        serialization failures are rejected with the expected errors.
    """
    obj = {'items':[1,2,'safe'], 'pair':(True,None)}
    encoded, size = F._object_to_tensor(obj)
    assert F._tensor_to_object(encoded, size) == obj
    padded, padded_size = F._object_to_tensor(obj, size+9)
    assert padded_size == size+9 and F._tensor_to_object(padded, size) == obj
    monkeypatch.setattr(F, 'SAFE_PICKLE', True)
    assert F.restricted_loads(pickle.dumps(set([1,2]))) == {1,2}
    with pytest.raises(pickle.UnpicklingError):
        F.restricted_loads(pickle.dumps(os.system))
    with pytest.raises(RuntimeError, match='serialize'):
        F._object_to_tensor(lambda: 0)
    with pytest.raises(RuntimeError, match='deserialize'):
        F._tensor_to_object(t([1,2], ms.int8), 2)
    F._check_all_tensors_or_tuple([t([1]), (1,2)])
    for value in (1, [1]):
        with pytest.raises(TypeError):
            F._check_all_tensors_or_tuple(value)
    F._check_all_tensor_same_dtype(t([1]), [None, t([2])])
    F._check_all_tensor_same_dtype_and_shape([None, t([1])], t([2]))
    with pytest.raises(TypeError):
        F._check_all_tensor_same_dtype(t([1]), t([2], ms.int32))
    for other in (t([1], ms.int32), t([1,2])):
        with pytest.raises(TypeError):
            F._check_all_tensor_same_dtype_and_shape(t([1]), other)
    with pytest.raises(TypeError):
        F._check_output_shape(t([1]), (2,), 'test')
    with pytest.raises(TypeError):
        F._check_output_dtype(t([1]), ms.int32, 'test')
    for op, tensor, peer, tag, error in [('bad', t([1]), 0, 0, TypeError),
            (object(),t([1]),0,0,TypeError), ('isend',1,0,0,TypeError),
            ('isend',t([1]),'0',0,TypeError), ('irecv',t([1]),0,1,NotImplementedError)]:
        with pytest.raises(error):
            F.P2POp(op, tensor, peer, tag=tag)
    monkeypatch.setattr(F, 'get_cache_group_size', lambda group: 2)
    with pytest.raises(ValueError):
        F._get_all_to_all_single_numel_list(t(np.ones((3,2))), t(np.ones((4,2))), None, None, 'g')
    for other in (t([1], ms.int32), t([1,2])):
        with pytest.raises(TypeError):
            F._check_tensor_list([other, other], t([1]), 2)


def test_comm_inplace_and_handles(monkeypatch):
    """
    Feature: Comm inplace and handles.
    Description: Check comm inplace and handles.
    Expectation: In-place policies, wait/synchronize ordering, and default handles match the explicit call
        sequences and unsupported configurations are rejected.
    """
    monkeypatch.setattr(F, '_COMM_ENABLE_PLACE', F._COMM_ENABLE_PLACE.copy())
    assert F.is_inplace_func('does-not-exist') is False
    F.set_comm_ops_inplace(False)
    assert all(value is False for value in F._COMM_ENABLE_PLACE.values())
    F.set_comm_ops_inplace(True, ['all_to_all_v_c'])
    assert F.is_inplace_func('all_to_all_v_c')
    for enabled, funcs, error in [(1,None,TypeError), (True,'bad',TypeError), (True,['bad'],ValueError)]:
        with pytest.raises(error):
            F.set_comm_ops_inplace(enabled, funcs)
    calls = []
    class Wait:
        def wait(self):
            calls.append('wait')
    monkeypatch.setattr(F, 'synchronize', lambda: calls.append('sync'))
    data, wait = t([2]), Wait()
    assert F._deal_comm_outputs((data,wait), False, True) == (data,None)
    assert calls == ['wait','sync']
    _, handle = F._deal_comm_outputs((data,wait), True, True)
    handle.wait()
    assert calls == ['wait','sync','wait','sync']
    assert F._deal_comm_outputs(data, False) == (data,None)
    assert F._deal_comm_outputs(data, True)[1] is F.default_handle
    assert F._deal_dist_comm_outputs((data,wait), False, True) == (data,None)
    assert F._deal_dist_comm_outputs(data, True)[1] is F.default_handle
    assert F._deal_dist_comm_outputs(data, False) == (data,None)


class Subject(SimpleNamespace):
    def get_attr_dict(self):
        return vars(self)


def subject(**kwargs):
    base = {
    'group': 'group',
    'group_back': 'back',
    'rank': 1,
    'rank_size': 2,
    'dev_num': 2,
    'mean_flag': False,
    'fusion': 7,
    'instance_name': 'forward',
    'parameter': 'weight',
    'op': O.ReduceOp.SUM,
    'do_mirror': True,
     'with_mirror_operator': True}
    base.update(kwargs)
    return Subject(**base)


class UnaryEndpoint:
    """One exact unary external communication endpoint and metadata capture."""
    def __init__(self, response):
        self.response = response
        self.calls = []
        self.payloads = []
        self.attrs = {}
        self.instance_name = None

    def __call__(self, value):
        self.calls.append(value)
        # Parameters may be assigned after communication; retain the actual send-time values.
        self.payloads.append(self.snapshot(value))
        return self.response

    @staticmethod
    def snapshot(value):
        if isinstance(value, Tensor):
            return value.dtype, value.asnumpy().copy()
        return tuple(UnaryEndpoint.snapshot(item) for item in value)

    def add_prim_attr(self, key, value):
        self.attrs[key] = value
        return self

    def set_prim_instance_name(self, name):
        self.instance_name = name
        return self


@pytest.fixture(name='grad_boundary')
def fixture_grad_boundary(monkeypatch):
    endpoints = []
    response = [t([8.,12.])]
    def all_reduce(op=O.ReduceOp.SUM, group=O.GlobalComm.WORLD_COMM_GROUP):
        endpoint = UnaryEndpoint(response[0])
        endpoints.append(('AllReduce', (op,group), endpoint))
        return endpoint
    def all_gather(group=O.GlobalComm.WORLD_COMM_GROUP):
        endpoint = UnaryEndpoint(response[0])
        endpoints.append(('AllGather', (group,), endpoint))
        return endpoint
    def reduce_scatter(op=O.ReduceOp.SUM, group=O.GlobalComm.WORLD_COMM_GROUP):
        endpoint = UnaryEndpoint(response[0])
        endpoints.append(('ReduceScatter', (op,group), endpoint))
        return endpoint
    monkeypatch.setattr(G, 'AllReduce', all_reduce)
    monkeypatch.setattr(G, 'AllGather', all_gather)
    monkeypatch.setattr(G, 'ReduceScatter', reduce_scatter)
    monkeypatch.setattr(G, '_get_enable_parallel_optimizer', lambda: False)
    monkeypatch.setattr(G, '_get_grad_accumulation_shard', lambda: False)
    monkeypatch.setattr(G, 'get_rank', lambda group=None: 1)
    monkeypatch.setattr(G, 'get_group_size', lambda group=None: 2)
    monkeypatch.setattr(G.ms, 'get_auto_parallel_context', lambda key: '' if key=='dump_local_norm_path' else False)
    monkeypatch.setattr(G._sdc_detector, 'need_sample', lambda: False)
    return endpoints, response


def assert_gradient_endpoint(endpoints, kind, arguments, attrs, name, payloads):
    """Require the complete outgoing protocol, including absence of extra communication."""
    matches = [entry for entry in endpoints if entry[0] == kind]
    assert len(matches) == 1
    _, actual_arguments, endpoint = matches[0]
    assert actual_arguments == arguments
    assert endpoint.attrs == attrs
    assert endpoint.instance_name == name
    assert len(endpoint.calls) == len(endpoint.payloads) == len(payloads)
    for (dtype, actual), expected in zip(endpoint.payloads, payloads):
        assert dtype == ms.float32
        np.testing.assert_array_equal(actual, expected)
    assert all(not other.calls for _, _, other in endpoints if other is not endpoint)


@pytest.mark.parametrize('op,expected', [('sum',[8,12]), ('prod',[4,4]), ('max',[0,12])])
def test_all_reduce_gradient_protocol_and_local_math(grad_boundary, op, expected):
    """
    Feature: All reduce gradient protocol and local math.
    Description: Check all reduce gradient protocol and local math.
    Expectation: AllReduce backward forwards the expected reduction/group/name and produces the independent
        sum/product gradient values.
    """
    endpoints, _ = grad_boundary
    x, out, dout = t([2,3]), t([4,3]), t([6,7])
    result = G.get_bprop_all_reduce(subject(op=op))(x,out,dout)
    check(result[0], expected)
    endpoint = endpoints[0][2]
    assert endpoints[0][:2] == ('AllReduce', ('sum','group'))
    assert endpoint.instance_name == 'gradforward'
    check(endpoint.calls[0], [24,21] if op=='prod' else [6,7])
    assert sum(len(item.calls) for _, _, item in endpoints) == 1


def test_local_virtual_gradients(grad_boundary):
    """
    Feature: Local virtual gradients.
    Description: Check local virtual gradients.
    Expectation: Virtual gradients preserve sensitivities and container structure, return explicit zero
        gradients where required, and avoid unnecessary communication.
    """
    x, dout = t([2.,3.]), t([4.,6.])
    a,b = G.get_bprop_virtual_add(None)(x,t([1.,2.]),None,dout)
    check(a,[5,8])
    check(b,[0,0])
    a,b = G.get_bprop_virtual_accu_grad(None)(x,t([7.,8.]),None,dout)
    check(a,[7,8])
    check(b,0)
    for dtype in (ms.bool_, ms.int16, ms.int32):
        d = t([1,0],dtype)
        assert G.get_bprop_virtual_div_operator(subject(divisor=2))(x,None,d)[0] is d
    div = G.get_bprop_virtual_div_operator(subject(divisor=2))
    check(div(x,None,dout)[0],[2,3])
    for values in ((dout,dout), [dout,dout]):
        result = div(x,None,values)[0]
        assert isinstance(result,type(values))
        for value in result:
            check(value,[2,3])
    assert G.get_bprop_broad_cast(None)(x,None,dout) == (dout,)
    check(G.get_bprop_get_tensor_slice_operator(None)(x,None,None,None,dout)[0],[0,0])
    y = Parameter(t([1.,2.]),name='accum')
    a,b = G.get_bprop_virtual_assign_add(subject(group=None))(x,y,None,dout)
    check(y,[5,8])
    check(a,0)
    check(b,0)
    for seq, expected in [(0,[0,0]), (1,[5,8])]:
        y = Parameter(t([1.,2.]),name='kv')
        values = G.get_bprop_virtual_assign_kv_cache(None)(x,y,t(seq,ms.int32),None,dout)
        check(values[0],[5,8])
        check(y,expected)
    # The exact empty-list result is part of this interface contract.
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    assert grad_boundary[0] == []


@pytest.mark.parametrize('mean', [False,True])
def test_all_gather_gradient_protocol(grad_boundary, mean):
    """
    Feature: All gather gradient protocol.
    Description: Check all gather gradient protocol.
    Expectation: Backward forwards the sensitivity and fusion metadata and applies the expected optional mean
        scaling; zero-size configurations are rejected.
    """
    endpoints,_ = grad_boundary
    dout=t([1.,2.,3.,4.])
    result=G.get_bprop_all_gather(subject(mean_flag=mean))(t([0,0]),None,dout)
    check(result[0],[4,6] if mean else [8,12])
    assert endpoints[0][2].calls == [dout]
    assert endpoints[0][2].attrs == {'fusion':7}
    assert_gradient_endpoint(endpoints, 'ReduceScatter', ('sum', 'group'), {'fusion': 7},
                             'grad_forward', [[1, 2, 3, 4]])
    with pytest.raises(ValueError,match='zero'):
        G.get_bprop_all_gather(subject(rank_size=0))


@pytest.mark.parametrize('mean,mirror,shard,add', [(a,b,c,d) for a in [False,True]
                         for b,c,d in [(True,False,False),(True,True,False),(True,True,True),(False,False,False)]])
def test_mini_all_gather_gradient(grad_boundary,monkeypatch,mean,mirror,shard,add):
    """
    Feature: Mini all gather gradient.
    Description: Check mini all gather gradient.
    Expectation: Mini all-gather backward returns the explicit mirrored/sharded gradients and applies the
        expected accumulator updates and zero gradients.
    """
    endpoints,response=grad_boundary
    response[0]=t([8.,12.,16.,20.])
    monkeypatch.setattr(G,'_get_grad_accumulation_shard',lambda:shard)
    z=Parameter(t([1,2] if shard else [1,2,3,4]), name='mini_accum')
    dout=t([3,4,5,6])
    dx,dz=G.get_bprop_mini_step_all_gather(subject(mean_flag=mean,do_mirror=mirror,add_accu=add))(t([0,0]),z,None,dout)
    check(dx,([8,10] if mean else [16,20]) if mirror else [3,4,5,6])
    check(dz,np.zeros(z.shape))
    assert_gradient_endpoint(endpoints, 'AllReduce', ('sum', 'group'), {'fusion': 7}, 'grad_forward',
                             ([[3, 4, 5, 6]] if shard else [[4, 6, 8, 10]]) if mirror else [])
    if mirror and not shard:
        check(z,[4,6,8,10])
    if mirror and shard and add:
        check(z,[9,12] if mean else [17,22])
    if not mirror or (shard and not add):
        check(z, [1, 2] if shard else [1, 2, 3, 4])


@pytest.mark.parametrize('mean,mirror',[(False,False),(True,False),(False,True),(True,True)])
def test_mirror_mini_gradient(grad_boundary,mean,mirror):
    """
    Feature: Mirror mini gradient.
    Description: Check mirror mini gradient.
    Expectation: Mini mirror backward honors mean/mirror branches, forwards fusion and parameter metadata, and
        updates accumulators to the literal expected values.
    """
    endpoints,_=grad_boundary
    z=Parameter(t([1,2]),name='mirror_accum')
    dout=t([3,4])
    dx,dz=G.get_bprop_mirror_mini_step_operator(subject(mean_flag=mean,do_mirror=mirror))(t([0,0]),z,None,dout)
    check(dx,([4,6] if mean else [8,12]) if mirror else ([1.5,2] if mean else [3,4]))
    check(dz,[0,0])
    if mirror:
        check(z,[4,6])
        assert endpoints[0][2].attrs=={'fusion':7,'parameter':'weight'}
    else:
        check(z, [1, 2])
    assert_gradient_endpoint(endpoints, 'AllReduce', ('sum', 'group'),
                             {'fusion': 7, 'parameter': 'weight'}, 'grad_mirrorforward',
                             [[4, 6]] if mirror else [])


@pytest.mark.parametrize('mean,with_mirror,mirror',[(a,b,c)
                         for a in [False,True] for b in [False,True] for c in [False,True]])
def test_micro_all_gather_gradient(grad_boundary,mean,with_mirror,mirror):
    """
    Feature: Micro all gather gradient.
    Description: Check micro all gather gradient.
    Expectation: Micro all-gather backward preserves segment/fusion metadata and returns the explicit
        gradient/accumulator values for each mirror branch.
    """
    endpoints,_=grad_boundary
    z=t([3,5])
    dout=t([1,2])
    dx,dz=G.get_bprop_micro_step_all_gather(
        subject(mean_flag=mean,with_mirror_operator=with_mirror,do_mirror=mirror,segment=4))(t([0,0]),z,None,dout)
    check(dx,([4,6] if mean else [8,12]) if mirror else ([1,2] if with_mirror else [3,5]))
    check(dz,1)
    if mirror:
        assert endpoints[0][2].attrs=={'fusion':7,'segment':4}
    if mirror:
        assert_gradient_endpoint(endpoints, 'ReduceScatter', ('sum', 'group'), {'fusion': 7, 'segment': 4},
                                 'grad_forward', [[1, 2]] if with_mirror else [[3, 5]])
    else:
        # The exact empty-list result is part of this interface contract.
        # pylint: disable-next=use-implicit-booleaness-not-comparison
        assert endpoints == []


@pytest.mark.parametrize('mean,dev_num',[(False,1),(False,2),(True,2)])
def test_mirror_dense_gradient(grad_boundary,mean,dev_num):
    """
    Feature: Mirror dense gradient.
    Description: Check mirror dense gradient.
    Expectation: Dense mirror gradients preserve values on one device and apply the explicit reduction and
        optional mean on multiple devices.
    """
    endpoints, _ = grad_boundary
    dout=t([3,4])
    dx,=G.get_bprop_mirror_operator(subject(mean_flag=mean,dev_num=dev_num))(t([0,0]),None,dout)
    check(dx,[3,4] if dev_num==1 else ([4,6] if mean else [8,12]))
    if dev_num == 1:
        # The exact empty-list result is part of this interface contract.
        # pylint: disable-next=use-implicit-booleaness-not-comparison
        assert endpoints == []
    else:
        assert_gradient_endpoint(endpoints, 'AllReduce', ('sum', 'group'),
                                 {'fusion': 7, 'parameter': 'weight'}, 'grad_mirrorforward', [[3, 4]])


@pytest.mark.parametrize('text', ['(1)', '(bad,2)', '(1,2', '1,2)'])
def test_log_invalid_vlog(text):
    """
    Feature: Log invalid vlog.
    Description: Check log invalid vlog.
    Expectation: Malformed verbose-log specifications return false and an empty parsed-level tuple.
    """
    assert L.parse_vlog_level(text) == (False,())


def test_log_validation_and_directories(monkeypatch,tmp_path):
    """
    Feature: Log validation and directories.
    Description: Check log validation and directories.
    Expectation: Valid log directories are created and returned; invalid log settings, paths, and permissions
        report the expected errors.
    """
    for bad in (None, 1, ' '):
        with pytest.raises(TypeError):
            L._make_directory(bad)
    assert L._make_directory(str(tmp_path)) == str(tmp_path)
    with pytest.raises(ValueError):
        L._check_directory_by_regular(1)
    for name in (None,'test'):
        with pytest.raises(ValueError):
            L._check_directory_by_regular('illegal space',prim_name=name)
    def denied(path,mode,exist_ok):
        raise PermissionError('injected')
    with monkeypatch.context() as patch:
        patch.setattr(L.os,'makedirs',denied)
        # _make_directory modifies process umask: isolate the real process state.
        patch.setattr(L.os,'umask',lambda mask: 0)
        with pytest.raises(TypeError,match='No write permission'):
            L._make_directory(str(tmp_path/'denied'))
    for cfg in ({'stderr_level':'8'}, {'level':'8'}, {'console':'bad'}, {'console':'0'},
                {'console':'0','filepath':str(tmp_path),'maxBytes':'bad'},
                {'console':'0','filepath':str(tmp_path),'backupCount':'bad'}):
        with pytest.raises(ValueError):
            L._verify_config(cfg)
    with monkeypatch.context() as patch:
        patch.setattr(L.os,'umask',lambda mask:0)
        L._verify_config({'console':'0','filepath':str(tmp_path/'created')})
    assert (tmp_path/'created').is_dir()


def test_log_formatter_rollover_and_config(monkeypatch,tmp_path):
    """
    Feature: Log formatter rollover and config.
    Description: Check log formatter rollover and config.
    Expectation: Log rollover preserves the previous record, formatting normalizes source paths, configuration
        retains explicit values, and handler cleanup empties the logger.
    """
    path=tmp_path/'runtime.log'
    handler=L._MultiCompatibleRotatingFileHandler(str(path),maxBytes=30,backupCount=2)
    try:
        handler.emit(logging.LogRecord('x',logging.INFO,'/outside/example.py',9,'first',(),None))
        handler.doRollover()
        assert (tmp_path/'runtime.log.1').read_text().strip()=='first'
        formatter=L._DataFormatter('UT','%(filepath)s %(message)s')
        record=logging.LogRecord('x',logging.INFO,'/outside/example.py',9,'message',(),None)
        assert formatter.format(record)=='/outside/example.py message'
        assert formatter.formatTime(record,'%Y').isdigit()
        record.pathname='/prefix/mindspore/foo.py'
        assert formatter.format(record)=='mindspore/foo.py message'
        logger=logging.Logger('runtime-comm-isolated',logging.INFO)
        handler.name='FileHandler'
        logger.addHandler(handler)
        stderr=logging.StreamHandler(io.StringIO())
        stderr.setLevel(logging.ERROR)
        logger.addHandler(stderr)
        monkeypatch.setattr(L,'GLOBAL_LOGGER',logger)
        cfg=L.get_log_config()
        assert cfg['GLOG_v']=='1' and cfg['GLOG_logtostderr']=='0'
        assert cfg['GLOG_log_dir']==str(tmp_path)
        assert cfg['logger_maxBytes']==30 and cfg['logger_backupCount']==2
        assert cfg['GLOG_stderrthreshold']=='3'
        assert L._setup_logger({}) is logger
        logger.removeHandler(handler)
        L._clear_handler(logger)
        # The exact empty-list result is part of this interface contract.
        # pylint: disable-next=use-implicit-booleaness-not-comparison
        assert logger.handlers==[]
        logger.addHandler(stderr)
        assert L.get_log_config()=={'GLOG_v':'1','GLOG_logtostderr':'1'}
    finally:
        handler.close()


@pytest.mark.parametrize('env,expected', [({},'0'), ({'MS_NODE_ID':'3'},'3'),
    ({'OMPI_COMM_WORLD_RANK':'2'},'2'), ({'RANK_ID':'1'},'1')])
def test_log_rank_env(monkeypatch,env,expected):
    """
    Feature: Log rank env.
    Description: Check log rank env.
    Expectation: Rank environment precedence resolves to the independently specified rank string.
    """
    for key in ('RANK_ID','OMPI_COMM_WORLD_RANK','MS_NODE_ID'):
        monkeypatch.delenv(key,raising=False)
    for key,value in env.items():
        monkeypatch.setenv(key,value)
    assert L._get_rank_id()==expected


def test_log_rank_conflict_and_stack(monkeypatch):
    """
    Feature: Log rank conflict and stack.
    Description: Check log rank conflict and stack.
    Expectation: Conflicting rank values emit the expected warning, caller/stack metadata is retained, and
        decorated functions preserve return values.
    """
    monkeypatch.setenv('RANK_ID','1')
    monkeypatch.setenv('OMPI_COMM_WORLD_RANK','2')
    with pytest.warns(UserWarning,match='different values'):
        assert L._get_rank_id()=='1'
    assert 'test_log_rank_conflict_and_stack' in L._get_stack_info(inspect.currentframe())
    assert L._find_caller(stack_info=True)[3].startswith('Stack (most recent call last):')
    action=L._LogActionOnce(SimpleNamespace(), 'runtime-comm')
    assert action(lambda x:x+1)(2)==3


def test_process_group_validation_and_boundary(monkeypatch):
    """
    Feature: Process group validation and boundary.
    Description: Check process group validation and boundary.
    Expectation: Initialization forwards backend, timeout, rank, size, and store as specified; invalid arguments
        and launcher exit behavior are preserved.
    """
    calls=[]
    store = object.__new__(F.TCPStore)  # Allocate inert Python wrapper without __init__/native resources.
    def init(backend):
        calls.append(('init',backend))
    def size(group=None):
        return 2
    def init_store(backend,timeout,world_size,rank,store,group_name):
        calls.append(('store',backend,timeout,world_size,rank,store,group_name))
    monkeypatch.setattr(F,'init',init)
    monkeypatch.setattr(F,'get_group_size',size)
    monkeypatch.setattr(F,'_init_with_store',init_store)
    monkeypatch.delenv('MS_ROLE',raising=False)
    F.init_process_group(world_size=2,pg_options=object(),device_id=1)
    assert calls==[('init','hccl')]
    for kwargs,error in [({'backend':'nccl'},ValueError),
            ({'init_method':'tcp://host:123','store':store},ValueError),
            ({'store':store,'world_size':0},ValueError),
            ({'store':store,'world_size':2,'rank':-1},ValueError),
            ({'init_method':'malformed','world_size':2,'rank':0},RuntimeError),
            ({'world_size':3},ValueError)]:
        with pytest.raises(error):
            F.init_process_group(**kwargs)
    F.init_process_group(store=store,world_size=2,rank=1,group_name='group')
    assert calls[-1][0:2]==('store','hccl') and calls[-1][3:]==(2,1,store,'group')
    assert calls[-1][2].total_seconds()==300
    monkeypatch.setenv('MS_ROLE','MS_SCHED')
    with pytest.raises(SystemExit) as error:
        F.init_process_group(store=store)
    assert error.value.code==0


def test_group_query_and_creation_failures(monkeypatch):
    """
    Feature: Group query and creation failures.
    Description: Check group query and creation failures.
    Expectation: Uninitialized queries return sentinel ranks, default groups retain identity, explicit ranks
        normalize in order, and invalid group arguments are rejected.
    """
    def failed(group):
        raise RuntimeError('injected uninitialized group')
    monkeypatch.setattr(F,'_get_rank_helper',failed)
    monkeypatch.setattr(F,'_get_size_helper',failed)
    assert F.get_rank()==-1 and F.get_world_size()==-1
    for fn in (F.get_rank,F.get_world_size,F.get_backend,F.destroy_process_group):
        with pytest.raises(TypeError):
            fn(1)
    assert F.new_group() == F.GlobalComm.WORLD_COMM_GROUP
    with pytest.raises(TypeError):
        F.new_group((0,1))
    with pytest.raises(TypeError):
        F.new_group([0,1],backend='bad')
    with pytest.raises(TypeError):
        F.new_group([0,1],pg_options=1)
    requested=[]
    def create(group,ranks,options):
        requested.append((group,ranks,options))
        raise RuntimeError('injected group failure')
    monkeypatch.setattr(F,'create_group',create)
    assert F.new_group([1,0],backend='hccl')==''
    assert requested[0][1:]==([0,1],None)
    assert requested[0][0].startswith('hccl_2_')


def test_all_to_all_v_c_request(monkeypatch):
    """
    Feature: All to all v c request.
    Description: Check all to all v c request.
    Expectation: Invalid collective counts and request arguments raise their specified validation errors before
        native communication.
    """
    monkeypatch.setattr(F,'_COMM_ENABLE_PLACE',F._COMM_ENABLE_PLACE.copy())
    F.set_comm_ops_inplace(True,['all_to_all_v_c'])
    monkeypatch.setattr(F,'get_cache_group_size',lambda group:2)
    monkeypatch.setattr(F,'get_cache_group_rank',lambda group:1)
    output=t([[0,0],[0,0]])
    input_=t([[1,2],[3,4]])
    for kwargs,error in [({'group':1},TypeError),({'async_op':1},TypeError),
            ({'send_count_matrix':(1,1,1,1)},TypeError),
            ({'send_count_matrix':[1,1,1,'a']},TypeError),
            ({'send_count_matrix':[1]},TypeError)]:
        args={'send_count_matrix': [1,1,1,1]}
        args.update(kwargs)
        with pytest.raises(error):
            F.all_to_all_v_c(output,input_,**args)
    F.set_comm_ops_inplace(False,['all_to_all_v_c'])
    with pytest.raises(ValueError):
        F.all_to_all_v_c(output,input_,[1,1,1,1])


def test_sparse_reduction_and_mirror_protocol(grad_boundary,monkeypatch):
    """
    Feature: Sparse reduction and mirror protocol.
    Description: Check sparse reduction and mirror protocol.
    Expectation: Sparse reductions forward indices and values separately and preserve the expected masked,
        mirrored, and optionally mean-scaled gradients.
    """
    endpoints,_=grad_boundary
    indices=t([0,1],ms.int32)
    values=t([[3,4],[5,6]])
    sparse=RowTensorInner(indices,values,(2,2))
    original=UnaryEndpoint.__call__
    current=[None]
    def gather_call(self,value):
        if self is current[0]:
            self.calls.append(value)
            return indices if value.dtype==ms.int32 else values
        return original(self,value)
    monkeypatch.setattr(UnaryEndpoint,'__call__',gather_call)
    for op in ('sum','max'):
        fn=G.get_bprop_all_reduce(subject(op=op))
        current[0]=endpoints[-1][2]
        dx,=fn(t([[1,2],[3,4]]),t([[0,2],[3,0]]),sparse)
        check(dx.indices,[0,1])
        check(dx.values,values.asnumpy() if op=='sum' else [[0,4],[5,0]])
        assert len(current[0].calls)==2
        check(current[0].calls[0],[0,1])
        check(current[0].calls[1],[[3,4],[5,6]])
    for mean in (False,True):
        fn=G.get_bprop_mirror_operator(subject(mean_flag=mean))
        current[0]=endpoints[-1][2]
        dx,=fn(t([[1,2],[3,4]]),None,sparse)
        check(dx.indices,[0,1])
        check(dx.values,values.asnumpy()/(2 if mean else 1))
    for mean in (False,True):
        z=Parameter(t([[1,2],[3,4]]),name='sparse_mini')
        dx,dz=G.get_bprop_mirror_mini_step_operator(subject(mean_flag=mean))(t([[1,2],[3,4]]),z,None,sparse)
        check(dx,np.zeros((2,2)))
        check(dz,np.zeros((2,2)))


def test_send_receive_gradient_requests(grad_boundary,monkeypatch):
    """
    Feature: Send receive gradient requests.
    Description: Check send receive gradient requests.
    Expectation: Backward send/receive requests preserve tags, ranks, groups, and ring metadata and return the
        explicit received or zero gradients.
    """
    requests=[]
    endpoints=[]
    def receive(
    sr_tag,
    src_rank,
    shape,
    dtype,
    group=O.GlobalComm.WORLD_COMM_GROUP,
    group_back=O.GlobalComm.WORLD_COMM_GROUP):
        endpoint=UnaryEndpoint(t([7,9],dtype))
        endpoints.append(endpoint)
        requests.append(('Receive',sr_tag,src_rank,shape,dtype,group,group_back))
        return endpoint
    def send(sr_tag,dest_rank,group=O.GlobalComm.WORLD_COMM_GROUP,group_back=O.GlobalComm.WORLD_COMM_GROUP):
        endpoint=UnaryEndpoint(t(0))
        endpoints.append(endpoint)
        requests.append(('Send',sr_tag,dest_rank,group,group_back))
        return endpoint
    monkeypatch.setattr(G,'Receive',receive)
    monkeypatch.setattr(G,'Send',send)
    s=subject(shape=[2],dtype=ms.float32,sr_tag=13,dst_global_rank=4,RING_ATTENTION_INDEX=2)
    dx,=G.get_bprop_send(s)(t([1,2]),None,t([3,4]))
    check(dx,[7,9])
    assert requests[-1]==('Receive',13,1,[2],ms.float32,'back',O.GlobalComm.WORLD_COMM_GROUP)
    assert endpoints[-1].attrs=={'src_global_rank':4,'RING_ATTENTION_INDEX':2}
    check(endpoints[-1].calls[0],0)
    for optimizer,flash in [(False,None),(True,None),(False,'True')]:
        monkeypatch.setattr(G,'_get_enable_parallel_optimizer',lambda optimizer=optimizer:optimizer)
        s=subject(shape=[2],sr_tag=13,src_global_rank=4,RING_ATTENTION_INDEX=2,flash_tag=flash)
        dout=t([3,4])
        dx,=G.get_bprop_receive(s)(t([1,2]),None,dout)
        check(dx,[0,0] if optimizer or flash=='True' else 0)
        assert endpoints[-1].calls==[dout]
        assert endpoints[-1].attrs=={'shape':[2],'dst_global_rank':4,'RING_ATTENTION_INDEX':2}
        assert requests[-1]==('Send',13,1,'back',O.GlobalComm.WORLD_COMM_GROUP)


def test_reduce_and_collective_gradient_requests(grad_boundary,monkeypatch):
    """
    Feature: Reduce and collective gradient requests.
    Description: Check reduce and collective gradient requests.
    Expectation: Reduce, gather, and scatter backward select the expected collective requests and preserve
        rank-dependent sensitivity values.
    """
    requests=[]
    endpoints=[]
    def broadcast(root_rank,group=O.GlobalComm.WORLD_COMM_GROUP):
        endpoint=UnaryEndpoint((t([10,20,30,40]),))
        endpoints.append(endpoint)
        requests.append(('Broadcast',root_rank,group))
        return endpoint
    def gather(dest_rank,group=O.GlobalComm.WORLD_COMM_GROUP):
        endpoint=UnaryEndpoint(t([10,20,30,40]))
        endpoints.append(endpoint)
        requests.append(('CollectiveGather',dest_rank,group))
        return endpoint
    monkeypatch.setattr(G,'Broadcast',broadcast)
    monkeypatch.setattr(G,'CollectiveGather',gather)
    dout=t([1,2,3,4])
    check(G.get_bprop_reduce(subject(dest_rank=0))(t([0,0,0,0]),None,dout)[0],[10,20,30,40])
    assert requests[-1]==('Broadcast',0,'group') and endpoints[-1].calls==[(dout,)]
    check(G.get_bprop_collective_gather(subject(dest_rank=0))(t([0,0]),None,dout)[0],[30,40])
    for rank in (0,1):
        monkeypatch.setattr(G,'get_rank',lambda group, rank=rank:rank)
        result=G.get_bprop_collective_scatter(subject(src_rank=0))(t([5,6,7,8]),None,dout)
        check(result[0],[10,20,30,40] if rank==0 else [0,0,0,0])
        assert requests[-1]==('CollectiveGather',0,'group')
        assert endpoints[-1].calls==[dout]


def test_host_and_scatter_gradient_requests(grad_boundary,monkeypatch):
    """
    Feature: Host and scatter gradient requests.
    Description: Check host and scatter gradient requests.
    Expectation: Host/scatter backward forwards the specified collective and gradient instance name, computes
        expected local values, and rejects invalid groups.
    """
    endpoints=[]
    requests=[]
    def host_gather(group):
        ep=UnaryEndpoint(t([9,10]))
        endpoints.append(ep)
        requests.append(('gather',group))
        return ep
    def host_scatter(op,group):
        ep=UnaryEndpoint(t([9,10]))
        endpoints.append(ep)
        requests.append(('scatter',op,group))
        return ep
    monkeypatch.setattr(G,'_HostAllGather',host_gather)
    monkeypatch.setattr(G,'_HostReduceScatter',host_scatter)
    dout=t([2,3])
    for fn,expected in [(G.get_bprop_host_all_gather,('scatter','sum',[0,1])),
            (G.get_bprop_host_reduce_scatter,('gather',[0,1]))]:
        check(fn(subject(group=[0,1]))(t([0,0]),None,dout)[0],[9,10])
        assert requests[-1]==expected and endpoints[-1].calls==[dout]
        assert endpoints[-1].instance_name=='gradforward'
    with pytest.raises(RuntimeError):
        G.get_bprop_host_reduce_scatter(subject(group=[0,1],op='max'))
    with pytest.raises(RuntimeError):
        G.get_bprop_reduce_scatter(subject(op='max'))
    check(G.get_bprop_reduce_scatter(subject())(t([0,0]),None,dout)[0],[8,12])
    with pytest.raises(RuntimeError):
        G.get_bprop_reduce_scatter_v(subject(op='max'))


class SizesEndpoint(UnaryEndpoint):
    def __call__(self,value,send_sizes,recv_sizes):
        self.calls.append((value,send_sizes,recv_sizes))
        return self.response


class MatrixEndpoint(UnaryEndpoint):
    def __call__(self,value,matrix):
        self.calls.append((value,matrix))
        return self.response


def test_variable_size_gradient_requests(grad_boundary,monkeypatch):
    """
    Feature: Variable size gradient requests.
    Description: Check variable size gradient requests.
    Expectation: Variable-size backward swaps send/receive metadata as specified and returns the expected
        sensitivity plus zero metadata gradients.
    """
    requests=[]
    endpoints=[]
    def swap(group):
        ep=SizesEndpoint(t([7,8]))
        endpoints.append(ep)
        requests.append(('swap',group))
        return ep
    def allv(group,block_size):
        ep=SizesEndpoint(t([7,8]))
        endpoints.append(ep)
        requests.append(('v',group,block_size))
        return ep
    def allvc(group,block_size,transpose=False):
        ep=MatrixEndpoint(t([7,8]))
        endpoints.append(ep)
        requests.append(('vc',group,block_size,transpose))
        return ep
    def gatherv(group):
        ep=MatrixEndpoint(t([7,8]))
        endpoints.append(ep)
        requests.append(('gatherv',group))
        return ep
    monkeypatch.setattr(G,'_AllSwap',swap)
    monkeypatch.setattr(G,'AlltoAllV',allv)
    monkeypatch.setattr(G,'AlltoAllVC',allvc)
    monkeypatch.setattr(G,'AllGatherV',gatherv)
    x=t([0,0])
    dout=t([1,2])
    send=t([1,3],ms.int64)
    recv=t([2,2],ms.int64)
    for fn,expected in [(G.get_bprop_allswap,('swap','group')),
            (G.get_bprop_all_to_all_v,('v','group',2))]:
        dx,ds,dr=fn(subject(block_size=2))(x,send,recv,None,dout)
        check(dx,[7,8])
        check(ds,[0,0])
        check(dr,[0,0])
        assert requests[-1]==expected and endpoints[-1].calls==[(dout,recv,send)]
        assert endpoints[-1].instance_name=='gradforward'
    for fn,expected in [(G.get_bprop_all_to_all_v_c,('vc','group',2,True)),
            (G.get_bprop_reduce_scatter_v,('gatherv','group'))]:
        dx,dm=fn(subject(block_size=2))(x,send,None,dout)
        check(dx,[7,8])
        check(dm,[0,0])
        assert requests[-1]==expected and endpoints[-1].calls==[(dout,send)]


def test_neighbor_gradient_requests(grad_boundary,monkeypatch):
    """
    Feature: Neighbor gradient requests.
    Description: Check neighbor gradient requests.
    Expectation: Neighbor backward swaps peer and shape metadata, forwards the original sensitivity once, and
        returns the supplied peer values.
    """
    requests=[]
    endpoints=[]
    def neighbor(send_rank_ids,recv_rank_ids,recv_shapes,send_shapes,recv_type,group=O.GlobalComm.WORLD_COMM_GROUP):
        ep=UnaryEndpoint(t([5,7]))
        endpoints.append(ep)
        requests.append((send_rank_ids,recv_rank_ids,recv_shapes,send_shapes,recv_type,group))
        return ep
    def neighborv2(send_rank_ids,send_lens,recv_rank_ids,recv_lens,data_format,group=O.GlobalComm.WORLD_COMM_GROUP):
        ep=UnaryEndpoint(t([5,7]))
        endpoints.append(ep)
        requests.append((send_rank_ids,send_lens,recv_rank_ids,recv_lens,data_format,group))
        return ep
    monkeypatch.setattr(G,'NeighborExchange',neighbor)
    monkeypatch.setattr(G.G,'NeighborExchangeV2Grad',neighborv2)
    s=subject(send_rank_ids=[0],recv_rank_ids=[1],send_shapes=((2,),),recv_shapes=((3,),),
              recv_type=ms.float32,send_lens=[1,2,3,4],recv_lens=[4,3,2,1],data_format='NCHW')
    dout=t([1,2])
    check(G.get_bprop_neighborexchange(s)(t([0,0]),None,dout)[0],[5,7])
    assert requests[-1]==([1],[0],((2,),),((3,),),ms.float32,'group')
    check(G.get_bprop_neighborexchangev2(s)(t([0,0]),None,dout)[0],[5,7])
    assert requests[-1]==([1],[4,3,2,1],[0],[1,2,3,4],'NCHW','group')
    assert all(ep.calls==[dout] for ep in endpoints)


def test_virtual_assign_communication_cast(grad_boundary):
    """
    Feature: Virtual assign communication cast.
    Description: Check virtual assign communication cast.
    Expectation: Virtual assignment forwards a float32 ReduceScatter request with the expected fusion/name
        metadata and updates the destination to [9, 14].
    """
    endpoints,_=grad_boundary
    y=Parameter(t([1,2]),name='shard_accum')
    result=G.get_bprop_virtual_assign_add(subject())(t([0,0]),y,None,t([3,4],ms.float16))
    check(y,[9,14])
    check(result[0],0)
    check(result[1],0)
    endpoint=endpoints[0][2]
    assert endpoints[0][:2]==('ReduceScatter',('sum','group'))
    assert endpoint.attrs=={'fusion':7,'not_delay_fusion':True}
    assert endpoint.instance_name=='_grad_accumulation_shard_gradforward'
    assert endpoint.calls[0].dtype==ms.float32
    check(endpoint.calls[0],[3,4])
    assert_gradient_endpoint(endpoints, 'ReduceScatter', ('sum', 'group'),
                             {'fusion': 7, 'not_delay_fusion': True},
                             '_grad_accumulation_shard_gradforward', [[3, 4]])


@pytest.mark.xfail(strict=True,raises=NameError,
                   reason='Product all_to_all_v_c refers to undefined size helper and communication endpoint')
def test_all_to_all_v_c_valid_input_missing_helper_reproducer(monkeypatch):
    """
    Feature: All to all v c valid input missing helper reproducer.
    Description: Check all to all v c valid input missing helper reproducer.
    Expectation: A valid request forwards the original output and converted source, world group, counts, size,
        rank, and synchronous flag exactly once.
    """
    monkeypatch.setattr(F,'_COMM_ENABLE_PLACE',F._COMM_ENABLE_PLACE.copy())
    F.set_comm_ops_inplace(True,['all_to_all_v_c'])
    monkeypatch.setattr(F,'get_cache_group_size',lambda group:2)
    monkeypatch.setattr(F,'get_cache_group_rank',lambda group:1)
    output, input_ = t([0, 0]), t([1, 2])
    requests = []
    def native(result, source, group, counts, size, rank, asynchronous):
        assert result is output
        check(source, [1, 2])
        assert source.dtype == input_.dtype
        assert (group, counts, size, rank, asynchronous) == (F.GlobalComm.WORLD_COMM_GROUP, [1, 1, 1, 1], 2, 1, False)
        requests.append(source)
        return output
    # Preserve the real missing Python helper. Isolate the external endpoint even after a future repair.
    monkeypatch.setattr(F, 'dist_comm_all_to_all_v_c_op', native, raising=False)
    # Correct expectation: a valid synchronous request returns no asynchronous handle.
    assert F.all_to_all_v_c(output, input_, [1, 1, 1, 1]) is None
    assert len(requests) == 1


@pytest.mark.parametrize('kind',['all_gather','mirror','micro_gather','micro_mirror'])
@pytest.mark.parametrize('dump_path',[False,True])
def test_gradient_norm_monitoring(grad_boundary,monkeypatch,tmp_path,capsys,kind,dump_path):
    """
    Feature: Gradient norm monitoring.
    Description: Check gradient norm monitoring.
    Expectation: Local norm accumulation remains non-trainable, records the expected squared norms and dump
        names, and preserves explicit gradient/parameter updates.
    """
    endpoints,response=grad_boundary
    response[0]=t([[8,12],[16,20]])
    monkeypatch.setattr(G,'_squared_device_local_norm',None)
    cfg={'dump_local_norm':True,'dump_device_local_norm':True,
         'dump_local_norm_path':str(tmp_path) if dump_path else ''}
    monkeypatch.setattr(G.ms,'get_auto_parallel_context',lambda key:cfg[key])
    monkeypatch.setattr(G._sdc_detector,'need_sample',lambda:True)
    monkeypatch.setattr(G._sdc_detector,'get_dump_name',lambda name:'sample:'+name)
    dumps=[]
    class Dump:
        def __call__(self,path,value):
            dumps.append((path,value))
            return t(0)
    monkeypatch.setattr(G.P,'TensorDump',Dump)
    source=t([[3,4],[0,0]])
    s=subject(mean_flag=True,mirror_user_id='weight',segment=5,parameter_micro=True,
              with_mirror_operator=False)
    if kind=='all_gather':
        dx,=G.get_bprop_all_gather(s)(t([[0,0],[0,0]]),None,source)
    elif kind=='mirror':
        dx,=G.get_bprop_mirror_operator(s)(t([[0,0],[0,0]]),None,source)
    elif kind=='micro_gather':
        dx,_=G.get_bprop_micro_step_all_gather(s)(t([[0,0],[0,0]]),source,None,t([[1,1],[1,1]]))
    else:
        z=Parameter(source,name='monitor_accum')
        result=G.get_bprop_mirror_micro_step_operator(s)(t([[0,0],[0,0]]),z,None,t([[1,1],[1,1]]))
        check(z,[[4,6],[8,10]])
        check(result[0],1)
        check(result[1],1)
        dx=z
    check(dx,[[4,6],[8,10]])
    check(G.get_squared_device_local_norm_param(),25)
    assert G.get_squared_device_local_norm_param().requires_grad is False
    assert dumps[-1][0]=='sample:weight'
    check(dumps[-1][1],16)
    if dump_path:
        assert dumps[0][0]==str(tmp_path/'rank_1'/'local_norm__weight')
        check(dumps[0][1],5)
    else:
        assert 'dump local norm:' in capsys.readouterr().out
    if kind in ('all_gather', 'micro_gather'):
        attrs = {'fusion': 7}
        if kind == 'micro_gather':
            attrs['segment'] = 5
        assert_gradient_endpoint(endpoints, 'ReduceScatter', ('sum', 'group'), attrs,
                                 'grad_forward', [[[3, 4], [0, 0]]])
    else:
        attrs = {'fusion': 7, 'parameter': 'weight'}
        if kind == 'micro_mirror':
            attrs['segment'] = 5
        assert_gradient_endpoint(endpoints, 'AllReduce', ('sum', 'group'), attrs,
                                 'grad_mirrorforward', [[[3, 4], [0, 0]]])


def test_squared_norm_disabled(grad_boundary,monkeypatch):
    """
    Feature: Squared norm disabled.
    Description: Check squared norm disabled.
    Expectation: Invalid squared norm disabled raise ValueError. Diagnostics identify False.
    """
    monkeypatch.setattr(G,'_squared_device_local_norm',None)
    with pytest.raises(ValueError,match='False'):
        G.get_squared_device_local_norm_param()


@pytest.mark.parametrize('dev_num,mean,opt_shard',[(1,False,False),(2,False,False),(2,True,True)])
def test_mirror_micro_assignment_and_optimizer(grad_boundary,monkeypatch,dev_num,mean,opt_shard):
    """
    Feature: Mirror micro assignment and optimizer.
    Description: Check mirror micro assignment and optimizer.
    Expectation: Micro mirror backward preserves fusion/segment/parameter metadata and applies the expected
        optimizer or assignment accumulator values.
    """
    endpoints,_=grad_boundary
    monkeypatch.setattr(G,'_get_enable_parallel_optimizer',lambda:opt_shard)
    z=Parameter(t([3,5]),name='micro_accum')
    dout=t([1,2])
    result=G.get_bprop_mirror_micro_step_operator(subject(dev_num=dev_num,mean_flag=mean,
                                                       segment=4,parameter_micro=True))(t([0,0]),z,None,dout)
    if opt_shard:
        check(result[0],[4,6])
        check(result[1],1)
        check(z,[3,5])
    else:
        check(result[0],1)
        check(result[1],1)
        check(z,[3,5] if dev_num==1 else [8,12])
    assert endpoints[0][2].attrs=={'segment':4,'fusion':7,'parameter':'weight'}
    assert_gradient_endpoint(endpoints, 'AllReduce', ('sum', 'group'),
                             {'segment': 4, 'fusion': 7, 'parameter': 'weight'},
                             'grad_mirrorforward', [[3, 5]] if dev_num > 1 else [])


@pytest.mark.parametrize('name,args',[
    ('all_gather_into_tensor_uneven',(None,None)), ('reduce_scatter_tensor_uneven',(None,None)),
    ('reduce',(None,0)), ('batch_isend_irecv',([],)), ('scatter_tensor',(None,None)),
    ('gather_into_tensor',(None,None)), ('broadcast',(None,0)), ('all_gather',([],None)),
    ('reduce_scatter',(None,[])), ('scatter',(None,[])), ('gather',(None,[])),
    ('scatter_object_list',([],[])), ('gather_object',(None,)),
    ('broadcast_object_list',([],)), ('all_gather_object',([],None)),
    ('send_object_list',([],0)), ('recv_object_list',([],)),
])
def test_public_comm_noninplace_guards(monkeypatch,name,args):
    """
    Feature: Public comm noninplace guards.
    Description: Check public comm noninplace guards.
    Expectation: Invalid public comm noninplace guards raise ValueError. Diagnostics identify Non-inplace.
    """
    monkeypatch.setattr(F,'_COMM_ENABLE_PLACE',F._COMM_ENABLE_PLACE.copy())
    F.set_comm_ops_inplace(False,[name])
    with pytest.raises(ValueError,match='Non-inplace'):
        getattr(F,name)(*args)


def test_comm_additional_argument_validation(monkeypatch):
    """
    Feature: Comm additional argument validation.
    Description: Check comm additional argument validation.
    Expectation: Invalid comm additional argument arguments raise TypeError.
    """
    monkeypatch.setattr(F,'get_group_rank_from_world_rank',lambda world_rank_id,group:world_rank_id)
    monkeypatch.setattr(F,'_COMM_ENABLE_PLACE',F._COMM_ENABLE_PLACE.copy())
    F.set_comm_ops_inplace(True)
    tensor=t([1,2])
    for kwargs in ({'op':1}, {'op':'bad'}, {'group':1}):
        with pytest.raises(TypeError):
            F.reduce_scatter_tensor_uneven(tensor,tensor,**kwargs)
    for tensor_,dst in [(1,0),(tensor,'0')]:
        with pytest.raises(TypeError):
            F.reduce(tensor_,dst)
    with pytest.raises(TypeError):
        F.gather_into_tensor(tensor,1)
    with pytest.raises(TypeError):
        F.gather_into_tensor(tensor,tensor,dst='0')
    with pytest.raises(TypeError):
        F.batch_isend_irecv([])
    with pytest.raises(TypeError):
        F.batch_isend_irecv([object()])
    first=F.P2POp('isend',tensor,1,group='a')
    second=F.P2POp('irecv',tensor,0,group='b')
    with pytest.raises(TypeError):
        F.batch_isend_irecv([first,second])
    second.group='a'
    second.op='bad'
    with pytest.raises(TypeError):
        F.batch_isend_irecv([second])
    second.op='irecv'
    second.tensor=1
    with pytest.raises(TypeError):
        F.batch_isend_irecv([second])


def test_batch_point_to_point_request(monkeypatch):
    """
    Feature: Batch point to point request.
    Description: Check batch point to point request.
    Expectation: Batched point-to-point dispatch forwards both tensors, world group, send/receive operation
        types, and remote ranks and returns the default handle.
    """
    monkeypatch.setattr(F,'_COMM_ENABLE_PLACE',F._COMM_ENABLE_PLACE.copy())
    F.set_comm_ops_inplace(True,['batch_isend_irecv'])
    calls=[]
    def endpoint(tensors,group,op_types,remote_ranks):
        calls.append((tensors,group,op_types,remote_ranks))
        return tensors
    monkeypatch.setattr(F,'dist_comm_batch_isend_irecv_op',endpoint)
    a,b=t([1,2]),t([3,4])
    result=F.batch_isend_irecv([F.P2POp(F.isend,a,1),F.P2POp(F.irecv,b,0)])
    assert result==[F.default_handle]
    assert calls[0][0]==[a,b] and calls[0][1:]==(F.GlobalComm.WORLD_COMM_GROUP,[F.TYPE_ISEND,F.TYPE_IRECV],[1,0])


def test_context_default_import_failure_restores_singletons(monkeypatch):
    """
    Feature: Context default import failure restores singletons.
    Description: Check context default import failure restores singletons.
    Expectation: Failure to import default configuration leaves one reusable context singleton with debug
        runtime enabled.
    """
    original=builtins.__import__
    def import_boundary(name,globals_=None,locals_=None,fromlist=(),level=0):
        if name=='mindspore' and fromlist==('default_config',):
            raise ImportError('injected default_config failure')
        return original(name,globals_,locals_,fromlist,level)
    monkeypatch.setattr(C,'K_CONTEXT',None)
    monkeypatch.setattr(C._Context,'_instance',None)
    monkeypatch.setattr(C,'MSContext',SimpleNamespace(get_instance=HandleBoundary))
    monkeypatch.setattr(builtins,'__import__',import_boundary)
    ctx=C._context()
    assert ctx is C._Context._instance and ctx.enable_debug_runtime is True
    assert C._context() is ctx


def test_context_deterministic_env_boundary(isolated_context,monkeypatch):
    """
    Feature: Context deterministic env boundary.
    Description: Check context deterministic env boundary.
    Expectation: Enabling determinism waits for communication initialization and stores ON plus the expected
        normalized environment values.
    """
    calls=[]
    class Collective:
        def wait_all_comm_init(self):
            calls.append('wait')
    monkeypatch.setattr(C,'CollectiveManager',SimpleNamespace(get_instance=Collective))
    monkeypatch.setenv('HCCL_DETERMINISTIC','invalid')
    monkeypatch.setenv('TE_PARALLEL_COMPILER','invalid')
    isolated_context.set_deterministic('ON')
    assert calls==['wait']
    assert os.environ['HCCL_DETERMINISTIC']=='true'
    assert os.environ['TE_PARALLEL_COMPILER']=='1'
    assert isolated_context.get_param(C.ms_ctx_param.deterministic)=='ON'


def test_context_public_attribute_and_enum_storage(isolated_context,monkeypatch):
    """
    Feature: Context public attribute and enum storage.
    Description: Check context public attribute and enum storage.
    Expectation: Public setters preserve the reserve-name flag, normalize boolean/integer graph-save options,
        and retain an empty GPU configuration.
    """
    monkeypatch.setattr(C,'K_CONTEXT',isolated_context)
    C.set_context(reserve_class_name_in_scope=False,save_graphs=False)
    assert isolated_context.reserve_class_name_in_scope is False
    assert isolated_context.get_param(C.ms_ctx_param.save_graphs)==0
    C.set_context(save_graphs=True)
    assert isolated_context.get_param(C.ms_ctx_param.save_graphs)==2
    # Real target-specific guard ignores GPU configuration on the CPU target.
    C.set_context(gpu_config={'conv_allow_tf32':True})
    assert isolated_context.gpu_config=={}


def test_remaining_local_helpers(monkeypatch,isolated_context,comm_environment):
    """
    Feature: Remaining local helpers.
    Description: Check remaining local helpers.
    Expectation: Restricted builtins, backend lookup, and caller metadata retain their expected values, while
        unsupported communication and shape requests are rejected.
    """
    with pytest.raises(TypeError):
        isolated_context._set_op_debug_option(1)
    for ctor,args in [(O.NeighborExchange,([0],[1],((2,),),((2,),),ms.float32)),
            (O.NeighborExchangeV2,([0],[1,1,1,1],[1],[1,1,1,1],'NCHW'))]:
        with pytest.raises(NotImplementedError):
            ctor(*args)(t([1,2]))
    assert F.RestrictedUnpickler(io.BytesIO()).find_class('builtins','range') is range
    monkeypatch.setattr(F,'_get_backend',lambda:'boundary-backend')
    assert F.get_backend('custom-group')=='boundary-backend'
    monkeypatch.setattr(F,'get_cache_group_size',lambda group:2)
    with pytest.raises(ValueError,match='output shape'):
        F._get_all_to_all_single_numel_list(t(np.ones((4,2))),t(np.ones((3,2))),None,None,'g')
    for value in (t([1,2],ms.int32),t([1,2,3])):
        with pytest.raises(TypeError):
            F._check_tensor_list([value,value],t([1,2]),2)
    def first():
        return second()
    def second():
        return third()
    def third():
        return L._find_caller()
    filename,line,name,stack=first()
    assert filename and line>0 and stack is None and name


def test_all_to_all_noninplace_reshape_protocol(monkeypatch):
    """
    Feature: All to all noninplace reshape protocol.
    Description: Check all to all noninplace reshape protocol.
    Expectation: Non-in-place all-to-all forwards packed values and sizes and reshapes peer results into the
        literal expected outputs with no asynchronous handle.
    """
    monkeypatch.setattr(F,'_COMM_ENABLE_PLACE',F._COMM_ENABLE_PLACE.copy())
    F.set_comm_ops_inplace(False,['all_to_all'])
    monkeypatch.setattr(F,'get_cache_group_size',lambda group:2)
    calls=[]
    def endpoint(input_,group,send_numel_list,recv_numel_list,rank_size,split_sizes_empty,async_op=True):
        calls.append((input_,group,send_numel_list,recv_numel_list,rank_size,split_sizes_empty,async_op))
        return t([10,20,30])
    monkeypatch.setattr(F,'inner_comm_all_to_all_v_op',endpoint)
    result,handle=F.all_to_all([(1,2),(1,)],[t([1,2]),t([3])])
    assert handle is None
    assert len(result)==2
    check(result[0],[[10,20]])
    check(result[1],[30])
    check(calls[0][0],[1,2,3])
    assert calls[0][1:]==(F.GlobalComm.WORLD_COMM_GROUP,[2,1],[2,1],2,False,False)


def test_mirror_micro_sparse_gradient(grad_boundary):
    """
    Feature: Mirror micro sparse gradient.
    Description: Check mirror micro sparse gradient.
    Expectation: Sparse micro mirror returns the specified scalar gradient sentinels and preserves the original
        destination parameter values.
    """
    sparse=RowTensorInner(t([0],ms.int32),t([[1,2]]),(2,2))
    z=Parameter(t([[3,4],[5,6]]),name='micro_sparse')
    result=G.get_bprop_mirror_micro_step_operator(subject())(t([[0,0],[0,0]]),z,None,sparse)
    check(result[0],1)
    check(result[1],1)
    check(z,[[3,4],[5,6]])


def test_store_scheduler_and_url_protocol_with_native_boundary(monkeypatch):
    """
    Feature: Store scheduler and url protocol with native boundary.
    Description: TCPStoreClient is replaced before any wrapper initialization; no socket is opened.
    Expectation: TCPStoreClient is replaced before any wrapper initialization; no socket is opened.
    """
    calls=[]
    def native_client(host_name,port,is_master,timeout_ms,world_size,wait_for_workers):
        calls.append((host_name,port,is_master,timeout_ms,world_size,wait_for_workers))
        return SimpleNamespace()
    monkeypatch.setattr(F,'TCPStoreClient',native_client)
    monkeypatch.setenv('MS_ROLE','MS_SCHED')
    inert=object.__new__(F.TCPStore)
    F.TCPStore.__init__(inert,'never-used',0)
    # Preserve the exact empty-list and uninitialized-wrapper contracts.
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    assert calls==[] and not hasattr(inert,'instance')
    monkeypatch.delenv('MS_ROLE')
    requests=[]
    def init_store(backend,timeout,world_size,rank,store,group_name):
        requests.append((backend,timeout,world_size,rank,store,group_name))
    monkeypatch.setattr(F,'_init_with_store',init_store)
    monkeypatch.setattr(F,'get_group_size',lambda group=None:2)
    F.init_process_group(init_method='tcp://127.0.0.1:12345',world_size=2,rank=0)
    assert calls==[('127.0.0.1',12345,True,300000,2,True)]
    assert requests[0][0]=='hccl' and requests[0][2:4]==(2,0)
    assert isinstance(requests[0][4],F.TCPStore)
    assert requests[0][4].host=='127.0.0.1' and requests[0][4].port==12345


@pytest.mark.xfail(strict=True,raises=AssertionError,
                   reason='Product _clear_handler removes while iterating and leaves every second handler')
def test_clear_all_log_handlers_reproducer():
    """
    Feature: Clear all log handlers reproducer.
    Description: Check clear all log handlers reproducer.
    Expectation: Handler clearing removes both registered handlers and leaves the logger with an empty handler
        list.
    """
    logger=logging.Logger('runtime-comm-clear-reproducer')
    first,second=logging.NullHandler(),logging.NullHandler()
    logger.addHandler(first)
    logger.addHandler(second)
    try:
        L._clear_handler(logger)
        # The exact empty-list result is part of this interface contract.
        # pylint: disable-next=use-implicit-booleaness-not-comparison
        assert logger.handlers==[]
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        first.close()
        second.close()
