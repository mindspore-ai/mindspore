# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Remaining runtime helpers: CPU numerics, explicit graph-body coverage and isolated native protocols."""
# Internal helper dispatch and state are the unit coverage targets in this module.
# pylint: disable=protected-access
import os
from types import SimpleNamespace

import numpy as np
import pytest
import mindspore as ms
from mindspore import amp
from mindspore import _checkparam as validator
from mindspore import device_manager as device
from mindspore.common import dtype
from mindspore.ops._utils import arg_handler
from mindspore.ops.composite import base, math_ops
from mindspore.ops.composite.multitype_ops import setitem_impl, add_impl, zeros_like_impl


@pytest.fixture(scope='module', autouse=True)
def cpu_context():
    old={key:ms.get_context(key) for key in ('mode','device_target','pynative_synchronize')}
    ms.set_context(mode=ms.PYNATIVE_MODE,device_target='CPU',pynative_synchronize=True)
    yield
    ms.set_context(**old)


@pytest.fixture(autouse=True)
def isolated_autograd_context_flag(monkeypatch):
    """The native not-convert-jit flag has no getter; isolate its setter instead of changing a singleton."""
    calls=[]
    def set_flag(value):
        assert value is True
        calls.append(value)
    handle=SimpleNamespace(_set_not_convert_jit=set_flag)
    monkeypatch.setattr(base,'MSContext',SimpleNamespace(get_instance=lambda:handle))
    yield calls


def t(value,kind=ms.float32):
    return ms.Tensor(value,kind)


def close(actual,expected):
    if isinstance(actual,(tuple,list)):
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected):
            close(a,b)
    else:
        assert isinstance(actual,ms.Tensor)
        assert actual.shape == np.asarray(expected).shape
        np.testing.assert_allclose(actual.asnumpy(),expected,rtol=2e-5,atol=2e-6)


@pytest.mark.parametrize('left,right,relation,expected',[
    (2,3,validator.NE,True),(2,2,validator.NE,False),
    (2,(1,3),validator.NOT_IN,True),(2,(1,2),validator.NOT_IN,False),
    (2,3,-999,False)])
def test_remaining_binary_relations(left,right,relation,expected):
    """
    Feature: Runtime validation, mixed precision and composite helpers
    Description: Check remaining binary relations.
    Expectation: Inequality and exclusion return the listed booleans; an unknown relation returns False.
    """
    assert validator._check_binary_rel(left,right,relation) is expected


def test_invalid_range_relation_and_message_formatters():
    """
    Feature: Runtime validation, mixed precision and composite helpers
    Description: Check invalid range relation and message formatters.
    Expectation: Unknown relations produce False or empty text; NE/NOT_IN messages and extrema match their values.
    """
    assert validator._check_inc_rel(2,1,3,-999) is False
    assert validator._format_str_one_value(3,validator.NE)=='!= 3'
    assert validator._format_str_one_value((1,2),validator.NOT_IN)=='not in (1, 2)'
    assert validator._format_str_one_value(3,-999)==''
    assert validator._format_str_two_value(1,3,-999)==''
    assert validator.max_(1,4,2)==4 and validator.min_(1,4,2)==1


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf')])
def test_check_is_number_rejects_nonfinite(value):
    """
    Feature: Runtime validation, mixed precision and composite helpers
    Description: Check check is number rejects nonfinite.
    Expectation: NaN and either infinity raise ValueError containing the legal-float diagnostic.
    """
    with pytest.raises(ValueError,match='legal float'):
        validator.check_is_number(value,float,'value','finite_check')


def test_subclass_python_type_identity_and_additional_message():
    """
    Feature: Runtime validation, mixed precision and composite helpers
    Description: Check subclass python type identity and additional message.
    Expectation: int is accepted among the declared classes; an invalid str class reports the supplied custom
        detail.
    """
    assert validator.check_subclass('kind',int,(float,int),'python_type') is None
    with pytest.raises(TypeError,match='custom detail'):
        validator.check_subclass('kind',str,(float,int),'python_type','custom detail')


@pytest.mark.parametrize('left,right,mix,error',[
    (dtype.float32,dtype.float32,False,None),
    (dtype.TensorType(dtype.float32),dtype.float32,True,None),
    (dtype.float32,dtype.TensorType(dtype.float32),True,None),
    (dtype.TensorType(dtype.float32),dtype.float32,False,TypeError),
    (dtype.float32,dtype.float64,False,TypeError),
    (dtype.int32,dtype.float32,True,TypeError)])
def test_scalar_tensor_dtype_mixing(left,right,mix,error):
    """
    Feature: Runtime validation, mixed precision and composite helpers
    Description: Check scalar tensor dtype mixing.
    Expectation: Compatible scalar/Tensor float dtypes return None; forbidden mixing or incompatible dtypes raise
        TypeError.
    """
    args={'left':left,'right':right}
    if error:
        with pytest.raises(error):
            validator.check_scalar_or_tensor_types_same(args,(dtype.float32,dtype.float64),'mix',allow_mix=mix)
    else:
        assert validator.check_scalar_or_tensor_types_same(
            args,(dtype.float32,dtype.float64),'mix',allow_mix=mix) is None


@pytest.mark.parametrize('name,args,error',[
    ('check_reduce_shape',((2,3),(2,),0,'reduce','input','output'),ValueError),
    ('check_astype_dtype',('imaginary128',),TypeError),
    ('check_astype_dtype',(None,),TypeError),
    ('check_transpose_axis',(('invalid',),2),TypeError),
    ('infer_out_shape',((2,3),(4,3)),ValueError)])
def test_remaining_shape_dtype_validation(name,args,error):
    """
    Feature: Runtime validation, mixed precision and composite helpers
    Description: Check remaining shape dtype validation.
    Expectation: Invalid reduction shapes, cast dtypes, transpose axes and broadcast shapes raise their declared
        errors.
    """
    with pytest.raises(error):
        getattr(validator,name)(*args)


def test_check_type_support_uses_device_specific_policy(monkeypatch):
    """
    Feature: Runtime validation, mixed precision and composite helpers
    Description: Check check type support uses device specific policy.
    Expectation: The active CPU enforces its dtype allowlist; a different target bypasses that check with exactly
        two queries.
    """
    calls=[]
    def get_context(key):
        assert key=='device_target'
        calls.append(key)
        return 'CPU'
    monkeypatch.setattr(validator,'context',SimpleNamespace(get_context=get_context))
    assert validator.check_type_support(dtype.float32,'CPU',(dtype.float32,)) is True
    assert validator.check_type_support(dtype.float64,'CPU',(dtype.float32,)) is False
    assert validator.check_type_support(dtype.float64,'GPU',(dtype.float32,)) is True
    assert calls==['device_target','device_target']


@pytest.mark.parametrize('name',['to_dilations','to_output_padding'])
@pytest.mark.parametrize('values',[(1,1,2,3),[1,1,2,3]])
def test_four_dimensional_convolution_arguments_select_spatial_values(name,values):
    """
    Feature: Runtime validation, mixed precision and composite helpers
    Description: Check four dimensional convolution arguments select spatial values.
    Expectation: Both tuple and list NCHW arguments produce the final two spatial values (2, 3).
    """
    assert getattr(arg_handler,name)('convolution','argument',values)==(2,3)


@pytest.mark.parametrize('scaler_class',[amp.StaticLossScaler,amp.DynamicLossScaler])
@pytest.mark.parametrize('container',['tensor','tuple','list'])
def test_loss_scaler_real_cpu_scale_unscale(scaler_class,container):
    """
    Feature: Runtime validation, mixed precision and composite helpers
    Description: Check loss scaler real CPU scale unscale.
    Expectation: Static and dynamic scalers multiply by eight and recover Tensor, tuple and list inputs; static
        adjustment is False.
    """
    scaler=scaler_class(8) if scaler_class is amp.StaticLossScaler else scaler_class(8,2,2)
    values=t([1.,-2.],ms.float16)
    inputs=values if container=='tensor' else (values,t([3.])) if container=='tuple' else [values,t([3.])]
    expected=[8.,-16.] if container=='tensor' else ([8.,-16.],[24.])
    # Public scaler methods execute the real @jit graph API on CPU.
    scaled=scaler.scale(inputs)
    close(scaled,expected)
    close(scaler.unscale(scaled),[1.,-2.] if container=='tensor' else ([1.,-2.],[3.]))
    if scaler_class is amp.StaticLossScaler:
        assert scaler.adjust(t(True,ms.bool_)) is False
        close(scaler.scale_value,8.)


@pytest.mark.parametrize('scaler_class',[amp.StaticLossScaler,amp.DynamicLossScaler])
@pytest.mark.parametrize('value,error',[(0,ValueError),(.5,ValueError),('eight',TypeError),(True,TypeError)])
def test_loss_scaler_initial_scale_validation(scaler_class,value,error):
    """
    Feature: Runtime validation, mixed precision and composite helpers
    Description: Check loss scaler initial scale validation.
    Expectation: Zero, fractional, textual and boolean scales raise the expected ValueError or TypeError.
    """
    with pytest.raises(error):
        if scaler_class is amp.StaticLossScaler:
            scaler_class(value)
        else:
            scaler_class(value,2,2)


@pytest.mark.parametrize('name',['scale','unscale','adjust'])
def test_loss_scaler_abstract_method_contract(name):
    """
    Feature: Runtime validation, mixed precision and composite helpers
    Description: Check loss scaler abstract method contract.
    Expectation: The base scale, unscale and adjust methods raise NotImplementedError.
    """
    with pytest.raises(NotImplementedError):
        getattr(amp.LossScaler,name)(None,t([1.]))


@pytest.mark.parametrize('bad',[False,True])
def test_amp_all_finite_public_graph_api_and_direct_body(bad):
    """
    Feature: Runtime validation, mixed precision and composite helpers
    Description: Check amp all finite public graph api and direct body.
    Expectation: Public and original graph bodies agree on finiteness; overflow returns zero or one for the same
        inputs.
    """
    values=(t([1.,np.inf] if bad else [1.,2.]),t([3.],ms.float16))
    close(amp.all_finite(values),not bad)
    # Explicit direct-body coverage is distinct from execution of the public compiled graph API.
    close(amp._all_finite.__wrapped__(values,None,False),not bad)
    close(amp._overflow(values[0]),int(bad))


def test_amp_scale_maps_direct_bodies_use_real_cpu_math():
    """
    Feature: Runtime validation, mixed precision and composite helpers
    Description: Check amp scale maps direct bodies use real CPU math.
    Expectation: Mixed-precision leaves are multiplied or divided by four using real CPU operations.
    """
    scale=t(4.)
    values=(t([1.,2.],ms.float16),t([3.]))
    close(amp._grad_scale_map.__wrapped__(scale,values),([4.,8.],[12.]))
    close(amp._grad_unscale_map.__wrapped__(scale,values),([.25,.5],[.75]))


@pytest.mark.parametrize('target,version,rank,conf,backend,jit,expected',[
    ('CPU',True,False,None,'',{},False),
    ('Ascend',False,False,None,'',{},False),
    ('Ascend',True,True,None,'',{},False),
    ('Ascend',True,False,'all_finite:true','',{},True),
    ('Ascend',True,False,'all_finite:True','',{},True),
    ('Ascend',True,False,'all_finite:false','',{},False),
    ('Ascend',True,False,'all_finite:False','',{},False),
    ('Ascend',True,False,None,'GE',{},False),
    ('Ascend',True,False,None,'',{'jit_level':'O0'},True),
    ('Ascend',True,False,None,'',{'jit_level':'O1'},True),
    ('Ascend',True,False,None,'',{'jit_level':'O2'},False),
    ('Ascend',True,False,None,'',{},True),
])
def test_amp_allfinite_enable_policy_isolated_runtime_queries(
    monkeypatch,target,version,rank,conf,backend,jit,expected):
    """
    Feature: Runtime validation, mixed precision and composite helpers
    Description: Policy only: native/environment queries are isolated without changing the actual device target.
    Expectation: Enablement matches target, version, rank, runtime override and JIT policy with the exact expected
        query order.
    """
    calls=[]
    def get_context(key):
        assert key=='device_target'
        calls.append(('target',key))
        return target
    def get_global_config():
        calls.append(('global_jit',))
        return jit
    class VersionBoundary:
        def __init__(self,library_path):
            assert library_path is None
            calls.append(('version_init',library_path))
        def check_custom_version(self):
            calls.append(('version_check',))
            return version
    class ExecutorBoundary:
        def get_jit_config(self):
            calls.append(('executor_jit',))
            return {'backend':backend}
    monkeypatch.setattr(amp,'context',SimpleNamespace(get_context=get_context,get_jit_config=get_global_config))
    monkeypatch.setattr(amp,'AscendEnvChecker',VersionBoundary)
    monkeypatch.setattr(amp,'JitExecutor_',SimpleNamespace(get_instance=ExecutorBoundary))
    for name,value in [('RANK_TABLE_FILE','isolated.json' if rank else None),('MS_DEV_RUNTIME_CONF',conf)]:
        if value is None:
            monkeypatch.delenv(name,raising=False)
        else:
            monkeypatch.setenv(name,value)
    assert amp._enable_all_finite() is expected
    expected_calls=[('target','device_target')]
    if target=='Ascend':
        expected_calls += [('version_init',None),('version_check',)]
        if version and not rank:
            expected_calls.append(('global_jit',))
            if conf is None:
                expected_calls.append(('executor_jit',))
    assert calls==expected_calls


class DeviceConfigurationBoundary:
    """Per-test replacement of one-time C++ device configuration; records every requested mutation."""
    def __init__(self):
        self.calls=[]
        self.enabled=False
        self.configured=False
        self.target='CPU'
        self.device_id=0
    def is_device_enable(self):
        self.calls.append(('is_device_enable',))
        return self.enabled
    def get_device_target(self):
        self.calls.append(('get_device_target',))
        return self.target
    def get_device_id(self):
        self.calls.append(('get_device_id',))
        return self.device_id
    def set_device(self,target,device_id,is_default):
        self.calls.append(('set_device',target,device_id,is_default))
    def is_deterministic_configured(self):
        self.calls.append(('is_deterministic_configured',))
        return self.configured
    def set_deterministic(self,value):
        self.calls.append(('set_deterministic',value))


@pytest.fixture(name='device_boundaries')
def fixture_device_boundaries(monkeypatch):
    configuration=DeviceConfigurationBoundary()
    calls=[]
    state={'initialized':False,'ps':False}
    def set_target(target):
        calls.append(('set_target',target))
    def get_device_context(target):
        calls.append(('get_device_context',target))
        return SimpleNamespace(initialized=lambda:state['initialized'])
    def wait():
        calls.append(('wait_all_comm_init',))
    def ps_reset(target):
        calls.append(('ps_reset',target))
        return state['ps']
    monkeypatch.setattr(device,'MSContext',SimpleNamespace(
        get_instance=lambda:SimpleNamespace(set_device_target_inner=set_target)))
    monkeypatch.setattr(device,'DeviceManagerConf',SimpleNamespace(get_instance=lambda:configuration))
    monkeypatch.setattr(device,'DeviceContextManager',SimpleNamespace(
        get_instance=lambda:SimpleNamespace(get_device_context=get_device_context)))
    monkeypatch.setattr(device,'CollectiveManager',SimpleNamespace(
        get_instance=lambda:SimpleNamespace(wait_all_comm_init=wait)))
    monkeypatch.setattr(device,'_need_reset_device_target_for_ps',ps_reset)
    return configuration,calls,state


@pytest.mark.parametrize('scenario',['invalid_target','negative_id','ps_reset',
                         'initialized','fresh','existing','conflict'])
def test_device_selection_native_protocol(device_boundaries,scenario):
    """
    Feature: Runtime validation, mixed precision and composite helpers
    Description: No actual device selection, native singleton mutation, or communication wait is performed.
    Expectation: Invalid or conflicting selections raise; fresh and existing selections issue exactly the expected
        native calls.
    """
    config,calls,state=device_boundaries
    if scenario=='invalid_target':
        with pytest.raises(ValueError,match='one of'):
            device.set_device('bad',0)
        assert calls==[] and config.calls==[]
    elif scenario=='negative_id':
        with pytest.raises(ValueError,match='equal to 0'):
            device.set_device('CPU',-1)
        assert calls==[('ps_reset','CPU')] and config.calls==[]
    else:
        if scenario=='ps_reset':
            state['ps']=True
        if scenario=='initialized':
            state['initialized']=True
        if scenario in ('existing','conflict'):
            config.enabled=True
        if scenario=='conflict':
            config.device_id=1
        if scenario in ('initialized','conflict'):
            with pytest.raises(RuntimeError,match='initialized|modified'):
                device.set_device('CPU',0)
        else:
            assert device.set_device('Ascend' if scenario=='ps_reset' else 'CPU') is None
        target='CPU'
        assert calls[:2]==[('ps_reset','Ascend' if scenario=='ps_reset' else 'CPU'),('set_target',target)]
        if scenario in ('fresh','ps_reset'):
            assert calls[-1]==('get_device_context','CPU')
            assert config.calls==[('is_device_enable',),('set_device','CPU',0,True)]
        elif scenario=='initialized':
            assert config.calls==[('is_device_enable',)]
        else:
            assert config.calls==[('is_device_enable',),('get_device_target',),('get_device_id',)]


@pytest.mark.parametrize('enabled,hccl,compiler,expected_hccl,expected_compiler',[
    (True,'strict','1','strict','1'),(True,'bad','bad','true','1'),
    (True,None,None,'true','1'),(False,'strict','1',None,None),
    (False,'false','0','false','0'),(False,None,None,None,None)])
def test_determinism_environment_policy_native_protocol(
    monkeypatch,device_boundaries,enabled,hccl,compiler,expected_hccl,expected_compiler):
    """
    Feature: Runtime validation, mixed precision and composite helpers
    Description: Check determinism environment policy native protocol.
    Expectation: HCCL/compiler variables follow enablement policy, native determinism is recorded, and original
        variables are restored.
    """
    config,calls,_=device_boundaries
    for key,value in [('HCCL_DETERMINISTIC',hccl),('TE_PARALLEL_COMPILER',compiler)]:
        if value is None:
            monkeypatch.delenv(key,raising=False)
        else:
            monkeypatch.setenv(key,value)
    # Restore the environment even when product code mutates/deletes these entries directly.
    with monkeypatch.context() as restore:
        for key in ('HCCL_DETERMINISTIC','TE_PARALLEL_COMPILER'):
            if key in os.environ:
                restore.setenv(key,os.environ[key])
            else:
                # delenv on an absent key records no undo. Register its absence before deletion.
                restore.setenv(key, '')
                del os.environ[key]
        device.set_deterministic(enabled)
        assert os.getenv('HCCL_DETERMINISTIC')==expected_hccl
        assert os.getenv('TE_PARALLEL_COMPILER')==expected_compiler
    assert os.getenv('HCCL_DETERMINISTIC') == hccl
    assert os.getenv('TE_PARALLEL_COMPILER') == compiler
    assert calls==[('wait_all_comm_init',)]
    assert config.calls==[('is_deterministic_configured',),('set_deterministic',enabled)]


@pytest.mark.parametrize('cast',[None,ms.float64])
def test_composite_matmul_real_cpu(cast):
    """
    Feature: Runtime validation, mixed precision and composite helpers
    Description: Check composite matmul real CPU.
    Expectation: Matrix values equal the NumPy product and the result uses the requested or default float dtype.
    """
    a=np.arange(6,dtype=np.float32).reshape(2,3)
    b=np.arange(6,dtype=np.float32).reshape(3,2)
    result=math_ops.matmul(t(a),t(b),dtype=cast)
    close(result,a@b)
    assert result.dtype==(ms.float32 if cast is None else cast)


@pytest.mark.parametrize('bad',[False,True])
def test_amp_allfinite_native_flag_boundary_and_real_boolean_inversion(monkeypatch,bad):
    """
    Feature: Runtime validation, mixed precision and composite helpers
    Description: Check amp allfinite native flag boundary and real boolean inversion.
    Expectation: The AllFinite primitive receives the original tuple once and its overflow boolean is inverted.
    """
    inputs=(t([1.,2.]),)
    calls=[]
    def native_status(primitive,received):
        assert primitive.name=='AllFinite' and received is inputs
        calls.append(received)
        return t(bad,ms.bool_)
    monkeypatch.setattr(amp.AllFinite,'__call__',native_status)
    close(amp._all_finite.__wrapped__(inputs,None,True),not bad)
    assert len(calls)==1


@pytest.mark.parametrize('bad',[False,True])
@pytest.mark.parametrize('target',['classic','saturation'])
def test_amp_npu_float_status_direct_body_protocol(monkeypatch,bad,target):
    """
    Feature: Runtime validation, mixed precision and composite helpers
    Description: Direct original graph body, simulated native feature/status boundaries; no Ascend runtime is used.
    Expectation: The original graph body reads and clears one eight-element int32 status in order and returns
        inverted overflow.
    """
    inputs=(t([1.,2.]),)
    calls=[]
    def get_context(key):
        assert key=='device_target'
        return 'Ascend'
    def feature(name):
        assert name in ('amp_loss_scale_train_classic_target','train_hq_dtype_line')
        calls.append(('feature',name))
        return (name=='amp_loss_scale_train_classic_target')==(target=='classic')
    monkeypatch.setattr(amp,'context',SimpleNamespace(get_context=get_context))
    monkeypatch.setattr(amp,'MSContext',SimpleNamespace(
        get_instance=lambda:SimpleNamespace(ascend_soc_supports_feature=feature)))
    status=t([int(bad)]+[0]*7,ms.int32)
    def get_status(primitive,received):
        assert primitive.name=='NPUGetFloatStatusV2'
        close(received,[0]*8)
        assert received.dtype==ms.int32
        calls.append(('get',received))
        return status
    def clear_status(primitive,received):
        assert primitive.name=='NPUClearFloatStatusV2'
        assert received is next(item[1] for item in calls if item[0]=='get')
        calls.append(('clear',received))
        return t([0]*8,ms.int32)
    monkeypatch.setattr(amp.NPUGetFloatStatusV2,'__call__',get_status)
    monkeypatch.setattr(amp.NPUClearFloatStatusV2,'__call__',clear_status)
    close(amp._all_finite.__wrapped__(inputs,'SATURATION_MODE',False),not bad)
    assert [item[0] for item in calls if item[0] in ('get','clear')]==['get','clear']


def test_amp_gpu_overflow_status_native_protocol(monkeypatch):
    """
    Feature: Runtime validation, mixed precision and composite helpers
    Description: Check amp GPU overflow status native protocol.
    Expectation: FloatStatus receives the original Tensor once and its exact output object is returned.
    """
    x=t([np.inf])
    output=t([1.])
    calls=[]
    def get_context(key):
        assert key=='device_target'
        return 'GPU'
    monkeypatch.setattr(amp,'context',SimpleNamespace(get_context=get_context))
    def native_call(primitive,received):
        assert primitive.name=='FloatStatus' and received is x
        calls.append(received)
        return output
    monkeypatch.setattr(amp.ops.FloatStatus,'__call__',native_call)
    assert amp._overflow(x) is output and len(calls)==1


@pytest.mark.parametrize('constructor,flag',[
    (base.GradOperation,'sens_param'),(base.GradOperation,'get_by_position'),
    (base._Grad,'get_by_position'),(base._Grad,'get_by_list'),(base._Grad,'sens_param'),
    (base._Grad,'get_value')])
def test_grad_boolean_flag_validation(constructor,flag):
    """
    Feature: Runtime validation, mixed precision and composite helpers
    Description: Check grad boolean flag validation.
    Expectation: Integer values for each boolean gradient configuration flag raise TypeError naming that flag.
    """
    with pytest.raises(TypeError,match=flag):
        constructor(**{flag:1})


def cubic_with_multiplier(x,y):
    return x*x*y


class WeightedQuadratic(ms.nn.Cell):
    def __init__(self):
        super().__init__()
        self.weight=ms.Parameter(t(2.),name='helper_weight')
    def construct(self,x):
        return self.weight*x*x


@pytest.mark.parametrize('constructor',[base.GradOperation,base._Grad])
@pytest.mark.parametrize('style',['position','weights','ordinary'])
def test_real_cpu_gradient_closures_and_graph_cache(constructor,style):
    """
    Feature: Runtime validation, mixed precision and composite helpers
    Description: Check real CPU gradient closures and graph cache.
    Expectation: Cached gradient closures retain identity; compiled and original bodies return the expected input or
        weight derivative.
    """
    previous=ms.get_context('mode')
    try:
        ms.set_context(mode=ms.GRAPH_MODE)
        if style=='position':
            operation=constructor(get_by_position=True)
            function=cubic_with_multiplier
            gradient=operation(function,None,(1,))
            expected=9.
            args=(t(3.),t(2.))
            repeat=operation(function,None,(1,))
        elif style=='weights':
            operation=constructor(get_by_list=True)
            function=WeightedQuadratic()
            weights=ms.ParameterTuple((function.weight,))
            gradient=operation(function,weights)
            expected=(9.,)
            args=(t(3.),)
            repeat=operation(function,weights)
        else:
            operation=constructor()
            function=cubic_with_multiplier
            gradient=operation(function)
            expected=12.
            args=(t(3.),t(2.))
            repeat=operation(function)
        assert repeat is gradient
        close(gradient(*args),expected)
        # The graph closure's original Python body is exercised explicitly in GRAPH_MODE,
        # while its inner gradient still executes the real compiled CPU graph API.
        close(gradient.__wrapped__(*args),expected)
    finally:
        ms.set_context(mode=previous)


@pytest.mark.parametrize('constructor',[base.GradOperation,base._Grad])
@pytest.mark.parametrize('inner',[False,True])
def test_gradient_rejected_inside_real_no_grad_context(constructor,inner):
    # Import the graph context at the point the executor state is inspected.
    """
    Feature: Runtime validation, mixed precision and composite helpers
    Description: Check gradient rejected inside real no grad context.
    Expectation: Both gradient factories reject no_grad execution and the executor gradient flag is restored
        afterward.
    """
    from mindspore.graph.api import _no_grad  # pylint: disable=import-outside-toplevel
    operation=constructor()
    operation.pynative_=inner
    before=base._pynative_executor.enable_grad()
    with _no_grad():
        with pytest.raises(RuntimeError,match='no_grad'):
            operation(cubic_with_multiplier)
    # The context manager must restore the executor's gradient flag.
    assert base._pynative_executor.enable_grad()==before


def test_gradient_forward_requires_callable():
    """
    Feature: Runtime validation, mixed precision and composite helpers
    Description: Check gradient forward requires callable.
    Expectation: Passing a numeric forward function raises TypeError identifying the callable requirement.
    """
    with pytest.raises(TypeError,match='callable'):
        base._pynative_func_grad_forward(3,t(1.))


@pytest.mark.parametrize('style',['ordinary','weights','position'])
def test_real_cpu_eager_gradient_closures(style):
    """
    Feature: Runtime validation, mixed precision and composite helpers
    Description: Check real CPU eager gradient closures.
    Expectation: Ordinary, weight and position gradients equal 12, 9 and 9; the position result remains a float32
        Tensor.
    """
    if style=='ordinary':
        close(base._Grad()(cubic_with_multiplier)(t(3.),t(2.)),12.)
    elif style=='weights':
        cell=WeightedQuadratic()
        close(base._Grad(get_by_list=True)(cell,ms.ParameterTuple((cell.weight,)))(t(3.)),(9.,))
    else:
        # Both the public grad factory and this internal factory return a Tensor for one position.
        result=base._Grad(get_by_position=True)(cubic_with_multiplier,None,(1,))(t(3.),t(2.))
        assert isinstance(result, ms.Tensor) and result.dtype == ms.float32
        close(result,9.)


def scalar_auxiliary_error(x):
    return x*x


def test_aux_gradient_requires_multiple_outputs():
    """
    Feature: Runtime validation, mixed precision and composite helpers
    Description: Check aux gradient requires multiple outputs.
    Expectation: A scalar-only forward result with has_aux enabled raises ValueError requiring multiple outputs.
    """
    with pytest.raises(ValueError,match='more than one'):
        base._Grad(has_aux=True)(scalar_auxiliary_error)(t(2.))


def test_gradient_output_identifiers_preserve_tensor_identity():
    """
    Feature: Runtime validation, mixed precision and composite helpers
    Description: Check gradient output identifiers preserve tensor identity.
    Expectation: Weight names accompany the original output Tensor; combining empty output tuples raises ValueError.
    """
    weight=ms.Parameter(t(2.),name='single_weight')
    output=t(6.)
    assert base._combine_weight(None,weight,output,[])==['single_weight',output]
    ids=base._combine_with_ids(None,weight,output)
    assert ids[0]=='single_weight' and ids[1] is output
    with pytest.raises(ValueError,match='empty tuple'):
        base._combine_with_ids(None,None,())


def test_multitype_graph_rejects_array_and_invalid_registration_type():
    """
    Feature: Runtime validation, mixed precision and composite helpers
    Description: Check multitype graph rejects array and invalid registration type.
    Expectation: NumPy input and invalid registration types raise TypeError without adding graph entries.
    """
    graph=base.MultitypeFuncGraph('runtime_helper_local')
    with pytest.raises(TypeError,match='numpy.ndarray'):
        graph(np.ones(2))
    with pytest.raises(TypeError,match='only support'):
        graph.register(3)(cubic_with_multiplier)
    # Require the exact empty registration list after both rejected operations.
    assert graph.entries==[]  # pylint: disable=use-implicit-booleaness-not-comparison


def test_map_with_bound_operation_real_cpu():
    """
    Feature: Runtime validation, mixed precision and composite helpers
    Description: Check map with bound operation real CPU.
    Expectation: The bound reverse Map squares every Tensor leaf while preserving tuple order.
    """
    square=base.MultitypeFuncGraph('runtime_helpers_square')
    @square.register('Tensor')
    def square_tensor(x):
        return x*x
    mapping=base.Map(square,reverse=True)
    close(mapping((t([1.,2.]),t([3.]))),([1.,4.],[9.]))


@pytest.mark.parametrize('name,index,value',[
    pytest.param(*case,marks=pytest.mark.xfail(strict=True,raises=TypeError,
                 reason=('boost setitem wrapper accepts sequence values but its Python fast helper '
                         'rejects list/tuple'))) if isinstance(case[2],(tuple,list)) else case
    for case in [
    ('_tensor_setitem_by_tuple_with_tuple',(slice(None),1),(8.,9.)),
    ('_tensor_setitem_by_tuple_with_list',(slice(None),1),[8.,9.]),
    ('_tensor_setitem_by_tensor_with_tuple','tensor',(7.,8.,9.)),
    ('_tensor_setitem_by_tensor_with_list','tensor',[7.,8.,9.]),
    ('_tensor_setitem_by_slice_with_list',slice(0,1),[7.,8.,9.]),
    ('_tensor_setitem_by_slice_with_tuple',slice(0,1),(7.,8.,9.)),
    ('_tensor_setitem_by_number_with_tuple',0,(7.,8.,9.)),
    ('_tensor_setitem_by_number_with_list',0,[7.,8.,9.]),
    ('_tensor_setitem_by_ellipsis_with_list',Ellipsis,[[7.,8.,9.],[10.,11.,12.]]),
    ('_tensor_setitem_by_ellipsis_with_tuple',Ellipsis,((7.,8.,9.),(10.,11.,12.))),
    ('_tensor_setitem_by_none_with_list',None,[[7.,8.,9.],[10.,11.,12.]]),
    ('_tensor_setitem_by_none_with_tuple',None,((7.,8.,9.),(10.,11.,12.))),
    ('_tensor_setitem_by_list_with_number',[0],7.),
    ('_tensor_setitem_by_list_with_tensor',[0],'tensor'),
    ('_tensor_setitem_by_list_with_tuple',[0],(7.,8.,9.)),
    ('_tensor_setitem_by_list_with_list',[0],[7.,8.,9.]),
]])
def test_setitem_remaining_numeric_wrappers(monkeypatch,name,index,value):
    """
    Feature: Runtime validation, mixed precision and composite helpers
    Description: Check setitem remaining numeric wrappers.
    Expectation: Updates match NumPy indexing or the exact IndexPut protocol; strict expected failures retain
        sequence-wrapper defects.
    """
    monkeypatch.setenv('MS_DEV_TENSOR_INDEX_BOOST','1')
    data=np.arange(6,dtype=np.float32).reshape(2,3)
    numpy_index=np.array([0],np.int32) if isinstance(index,str) else index
    actual_index=t([0],ms.int32) if isinstance(index,str) else index
    oracle_value=np.array([7.,8.,9.],np.float32) if isinstance(value,str) else value
    actual_value=t(oracle_value) if isinstance(value,str) else value
    expected=data.copy()
    expected[numpy_index]=oracle_value
    if isinstance(value,(tuple,list)):
        close(getattr(setitem_impl,name)(t(data),actual_index,actual_value),expected)
    else:
        calls=[]
        input_x=t(data)
        def index_put(destination,indices,updates,accumulate):
            assert destination is input_x and isinstance(indices,tuple) and len(indices)==1
            close(indices[0],[0])
            assert indices[0].dtype in (ms.int32,ms.int64)
            close(updates,oracle_value)
            assert updates.dtype==ms.float32 and accumulate is False
            calls.append((destination,indices,updates,accumulate))
            return destination
        monkeypatch.setattr(setitem_impl.compile_utils,'inplace_index_put_op',index_put)
        assert getattr(setitem_impl,name)(input_x,actual_index,actual_value) is input_x
        assert len(calls)==1



def square_for_transform(x):
    return x*x


def test_taylor_real_cpu_graph_cache_and_original_body():
    """
    Feature: Runtime validation, mixed precision and composite helpers
    Description: Check taylor real CPU graph cache and original body.
    Expectation: The cached Taylor transform retains identity and both bodies produce coefficients [4, 4, 2].
    """
    operation=base._TaylorOperation()
    transformed=operation(square_for_transform)
    assert operation(square_for_transform) is transformed
    values=t([2.,1.,0.])
    close(transformed(values),[4.,4.,2.])
    close(transformed.__wrapped__(values),[4.,4.,2.])


def test_vmap_real_cpu_graph_api_and_original_body():
    """
    Feature: Runtime validation, mixed precision and composite helpers
    Description: Check vmap real CPU graph api and original body.
    Expectation: The cached vmap closure retains identity and both bodies square [1, 2, 3] into [1, 4, 9].
    """
    operation=base._Vmap()
    mapped=operation(square_for_transform,0,0)
    assert operation(square_for_transform,0,0) is mapped
    close(mapped(t([1.,2.,3.])),[1.,4.,9.])
    close(mapped.__wrapped__(t([1.,2.,3.])),[1.,4.,9.])


@pytest.mark.parametrize('kind',['UMonad','IOMonad'])
def test_real_monad_identity_contracts(kind):
    # Instantiate native monads only in the identity-contract test.
    """
    Feature: Runtime validation, mixed precision and composite helpers
    Description: Check real monad identity contracts.
    Expectation: Zero-like and addition of real UMonad/IOMonad values return the original left object.
    """
    from mindspore import _c_expression as native  # pylint: disable=import-outside-toplevel
    left,right=getattr(native,kind)(),getattr(native,kind)()
    zeros_fn=zeros_like_impl._zeros_like_u_monad if kind=='UMonad' else zeros_like_impl._zeros_like_io_monad
    add_fn=add_impl._add_umonad_umonad if kind=='UMonad' else add_impl._add_iomonad_iomonad
    assert zeros_fn(left) is left
    assert add_fn(left,right) is left


@pytest.fixture(name='actual_map_tensor')
def fixture_actual_map_tensor():
    # Keep experimental native storage construction within fixture setup.
    from mindspore.experimental import MapParameter  # pylint: disable=import-outside-toplevel
    return MapParameter(key_tensor=t([1],ms.int32),value_tensor=t([[2.,3.]]),default_value='zeros')._map_tensor


def test_map_tensor_backprop_identity_contracts(actual_map_tensor):
    """
    Feature: Runtime validation, mixed precision and composite helpers
    Description: Check map tensor backprop identity contracts.
    Expectation: Zero-like and backward addition preserve the exact MapTensor object.
    """
    assert zeros_like_impl._zeros_like_map_tensor(actual_map_tensor) is actual_map_tensor
    assert add_impl._map_tensor_add_backward(actual_map_tensor,actual_map_tensor) is actual_map_tensor


def test_map_setitem_native_put_protocol(monkeypatch,actual_map_tensor):
    """
    Feature: Runtime validation, mixed precision and composite helpers
    Description: Check map setitem native put protocol.
    Expectation: Put receives the original MapTensor, key Tensor and value Tensor once and returns that MapTensor.
    """
    keys,values=t([1,3],ms.int32),t([[5.,6.],[7.,8.]])
    calls=[]
    def put(mapping,k,v):
        assert mapping is actual_map_tensor and k is keys and v is values
        calls.append((mapping,k,v))
        return mapping
    monkeypatch.setattr(setitem_impl._map_tensor_ops,'put',put)
    assert setitem_impl._map_tensor_setitem(actual_map_tensor,keys,values) is actual_map_tensor
    assert len(calls)==1


def test_zeros_like_dictionary_real_leaves_native_make_dict_protocol(monkeypatch):
    """
    Feature: Runtime validation, mixed precision and composite helpers
    Description: Only MakeDict construction is a native boundary; recursive leaf zero tensors are real CPU
        operations.
    Expectation: MakeDict receives ordered keys and recursively zeroed leaves once, preserving numeric leaf
        categories.
    """
    values={'weight':t([[2.,3.]]),'count':3,'rate':2.5}
    calls=[]
    def make_dict(keys,leaves):
        assert list(keys)==['weight','count','rate']
        close(leaves[0],[[0.,0.]])
        assert leaves[1:]==(0,0.)
        calls.append((keys,leaves))
        return dict(zip(keys,leaves))
    monkeypatch.setattr(zeros_like_impl.F,'make_dict',make_dict)
    result=zeros_like_impl._zeros_like_dict(values)
    assert list(result)==list(values) and len(calls)==1
    close(result['weight'],[[0.,0.]])
    assert result['count']==0 and result['rate']==0. and isinstance(result['rate'],float)


def test_row_tensor_add_zero_native_protocol(monkeypatch):
    # Native sparse storage is needed only for this protocol case.
    """
    Feature: Runtime validation, mixed precision and composite helpers
    Description: Check row tensor add zero native protocol.
    Expectation: The native row-add boundary receives the original sparse and dense operands once and preserves
        sparse values.
    """
    from mindspore.common.sparse_tensor import RowTensorInner  # pylint: disable=import-outside-toplevel
    row=RowTensorInner(t([0],ms.int32),t([[2.,3.]]),(3,2))
    dense=t(np.zeros((3,2)))
    calls=[]
    def native_call(sparse,other):
        assert sparse is row and other is dense
        calls.append((sparse,other))
        return row
    monkeypatch.setattr(add_impl.F,'row_tensor_add',native_call)
    assert add_impl.add_rowtensor_tensor(row,dense) is row and len(calls)==1
    close(row.values,[[2.,3.]])


@pytest.mark.xfail(strict=True,raises=AttributeError,
                   reason=('tuple + COO backward helper misspells check_coo_tensor_input_length '
                           'as heck_coo_tensor_input_length'))
def test_tuple_cootensor_addition_has_correct_sparse_sum():
    """
    Feature: Runtime validation, mixed precision and composite helpers
    Description: Check tuple cootensor addition has correct sparse sum.
    Expectation: The sparse sum should contain both coordinates; strict xfail exposes the misspelled validation
        helper.
    """
    sparse=ms.COOTensor(t([[1,1]],ms.int32),t([3.]),(2,2))
    components=(t([[0,0]],ms.int32),t([2.]))
    result=add_impl._add_tuple_cootensor(components,sparse)
    close(result.to_dense(),[[2.,0.],[0.,3.]])


@pytest.mark.xfail(strict=True,raises=TypeError,
                   reason=('backward addition registered for None,None evaluates None + None '
                           'instead of preserving the absent gradient'))
def test_backward_absent_gradients_remain_none():
    """
    Feature: Runtime validation, mixed precision and composite helpers
    Description: Check backward absent gradients remain none.
    Expectation: Adding two absent gradients should return None; strict xfail records the current invalid None
        addition.
    """
    assert add_impl._add_nonetensor_tensor(None,None) is None


@pytest.mark.parametrize('kind',['list','dict'])
def test_sequence_setitem_native_boundary_protocol(monkeypatch,kind):
    """
    Feature: Runtime validation, mixed precision and composite helpers
    Description: Check sequence setitem native boundary protocol.
    Expectation: The selected list/dict boundary receives the original arguments once and returns the exact
        replacement container.
    """
    values=[t([1.]),t([2.])] if kind=='list' else {'a':t([1.])}
    replacement=t([3.])
    index=1 if kind=='list' else 'b'
    calls=[]
    output=[values[0],replacement] if kind=='list' else {'a':values['a'],'b':replacement}
    def native_call(data,key,value):
        assert data is values and key==index and value is replacement
        calls.append((data,key,value))
        return output
    if kind=='list':
        monkeypatch.setattr(setitem_impl.F,'list_setitem',native_call)
        result=setitem_impl._list_setitem_with_tensor(values,index,replacement)
    else:
        monkeypatch.setattr(setitem_impl,'_dict_setitem',native_call)
        result=setitem_impl._dict_setitem_with_tensor(values,index,replacement)
    assert result is output and len(calls)==1
    assert result[index] is replacement


@pytest.mark.xfail(strict=True,raises=AssertionError,
                   reason=('zeros_like_leaf dispatch matches Bool as Number first and returns '
                           'integer 0 instead of bool False'))
def test_zeros_like_boolean_preserves_boolean_type():
    """
    Feature: Runtime validation, mixed precision and composite helpers
    Description: Check zeros like boolean preserves boolean type.
    Expectation: Boolean zero-like should be False by identity; strict xfail records dispatch returning integer
        zero.
    """
    assert zeros_like_impl.zeros_like_leaf(True) is False
