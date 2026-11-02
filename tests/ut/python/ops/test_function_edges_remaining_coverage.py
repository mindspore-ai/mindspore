# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Final function edges: genuine CPU values and precisely bounded sparse native protocols."""
import numpy as np
import pytest
import mindspore as ms
from mindspore.ops.function import sparse_func as sparse
from mindspore.ops.function import nn_func as nn
from mindspore.ops.function import array_func as array


@pytest.fixture(scope='module',autouse=True)
def cpu_context():
    previous={key:ms.get_context(key) for key in ('mode','device_target','pynative_synchronize')}
    ms.set_context(mode=ms.PYNATIVE_MODE,device_target='CPU',pynative_synchronize=True)
    yield
    ms.set_context(**previous)


def t(value,dtype=ms.float32):return ms.Tensor(value,dtype)


def tensor_fields(value,expected,dtype):
    assert isinstance(value,ms.Tensor) and value.dtype==dtype
    expected=np.asarray(expected)
    assert value.shape==expected.shape
    np.testing.assert_allclose(value.asnumpy(),expected,rtol=1e-6,atol=1e-6)


def make_csr(index_dtype=ms.int32,value_dtype=ms.float32):
    return ms.CSRTensor(t([0,2,3],index_dtype),t([0,2,1],index_dtype),t([1,2,3],value_dtype),(2,3))


def csr_fields(value,indptr,indices,values,shape,index_dtype,value_dtype):
    assert isinstance(value,ms.CSRTensor) and value.shape==shape
    tensor_fields(value.indptr,indptr,index_dtype)
    tensor_fields(value.indices,indices,index_dtype)
    tensor_fields(value.values,values,value_dtype)


@pytest.mark.parametrize('operation',['csr_div','csr_mul'])
@pytest.mark.parametrize('index_dtype',[ms.int32,ms.int64])
@pytest.mark.parametrize('value_dtype,dense_dtype,promoted',[
    (ms.float16,ms.float32,ms.float32),(ms.float32,ms.float64,ms.float64)])
@pytest.mark.parametrize('broadcast',['row','column'])
def test_csr_dense_arithmetic_native_protocol(monkeypatch,operation,index_dtype,value_dtype,dense_dtype,promoted,broadcast):
    """Protocol only: real promotion and CSR assembly; supplied native values are not kernel evidence."""
    x=make_csr(index_dtype,value_dtype)
    dense=np.array([[2.,3.,4.]]) if broadcast=='row' else np.array([[2.],[4.]])
    y=t(dense,dense_dtype)
    selected=np.array([2.,4.,3.]) if broadcast=='row' else np.array([2.,2.,4.])
    supplied_values=np.array([1.,2.,3.])/selected if operation=='csr_div' else np.array([1.,2.,3.])*selected
    native_output=t(supplied_values,promoted);calls=[]
    primitive_class=sparse._csr_ops.CSRDiv if operation=='csr_div' else sparse._csr_ops.CSRMul
    def native_call(primitive,indptr,indices,values,shape,dense_arg):
        assert primitive.name==('CSRDiv' if operation=='csr_div' else 'CSRMul')
        tensor_fields(indptr,[0,2,3],index_dtype)
        tensor_fields(indices,[0,2,1],index_dtype)
        tensor_fields(values,[1.,2.,3.],promoted)
        assert shape==(2,3)
        tensor_fields(dense_arg,dense,promoted)
        calls.append((indptr,indices,values,shape,dense_arg))
        return native_output
    monkeypatch.setattr(primitive_class,'__call__',native_call)
    result=getattr(sparse,operation)(x,y)
    assert len(calls)==1
    csr_fields(result,[0,2,3],[0,2,1],supplied_values,(2,3),index_dtype,promoted)


@pytest.mark.parametrize('index_dtype',[ms.int32,ms.int64])
@pytest.mark.parametrize('value_dtype,dense_dtype,promoted',[
    (ms.float16,ms.float32,ms.float32),(ms.float32,ms.float64,ms.float64)])
def test_csr_matrix_vector_native_protocol(monkeypatch,index_dtype,value_dtype,dense_dtype,promoted):
    """Protocol only for the CPU build's unavailable LLVM CSR kernel."""
    x=make_csr(index_dtype,value_dtype);vector=t([[2.],[4.],[6.]],dense_dtype)
    native_output=t([[14.],[12.]],promoted);calls=[]
    def native_call(primitive,indptr,indices,values,shape,dense):
        assert primitive.name=='CSRMV' and shape==(2,3)
        tensor_fields(indptr,[0,2,3],index_dtype)
        tensor_fields(indices,[0,2,1],index_dtype)
        tensor_fields(values,[1.,2.,3.],promoted)
        tensor_fields(dense,[[2.],[4.],[6.]],promoted)
        calls.append((indptr,indices,values,shape,dense));return native_output
    monkeypatch.setattr(sparse._csr_ops.CSRMV,'__call__',native_call)
    assert sparse.csr_mv(x,vector) is native_output
    assert len(calls)==1
    tensor_fields(native_output,[[14.],[12.]],promoted)


def csr_from_array(values,index_dtype):
    values=np.asarray(values)
    rows,columns=np.nonzero(values)
    indptr=np.concatenate([[0],np.cumsum(np.bincount(rows,minlength=values.shape[0]))])
    value_dtype=ms.complex64 if np.iscomplexobj(values) else ms.float32
    result=ms.CSRTensor(t(indptr,index_dtype),t(columns,index_dtype),t(values[rows,columns],value_dtype),values.shape)
    return result,indptr,columns,values[rows,columns],value_dtype


@pytest.mark.parametrize('index_dtype',[ms.int32,ms.int64])
@pytest.mark.parametrize('trans_a,trans_b,adjoint_a,adjoint_b',[
    (False,False,False,False),(True,False,False,False),(False,True,False,False),
    (False,False,True,False),(False,False,False,True),
    pytest.param(True,True,False,False,marks=pytest.mark.xfail(
        strict=True,raises=(AssertionError,ValueError),
        reason='csr_mm returns b.shape[1] instead of b.shape[0] when both inputs are transposed'))])
def test_sparse_matrix_product_native_protocol(monkeypatch,index_dtype,trans_a,trans_b,adjoint_a,adjoint_b):
    """Protocol only for GPU sparse matmul; all ten input fields and four flags are checked."""
    a_dense=np.array([[1.,0.,2.],[0.,3.,4.]],np.float32)
    b_dense=np.array([[2.,0.,1.,0.],[0.,3.,0.,2.],[4.,0.,2.,1.]],np.float32)
    if adjoint_a or adjoint_b:
        a_dense=a_dense.astype(np.complex64)*(1+1j)
        b_dense=b_dense.astype(np.complex64)*(1-2j)
    a_stored=a_dense.conj().T if adjoint_a else a_dense.T if trans_a else a_dense
    b_stored=b_dense.conj().T if adjoint_b else b_dense.T if trans_b else b_dense
    a,a_ptr,a_col,a_val,value_dtype=csr_from_array(a_stored,index_dtype)
    b,b_ptr,b_col,b_val,_=csr_from_array(b_stored,index_dtype)
    supplied_product=a_dense@b_dense
    supplied,c_ptr,c_col,c_val,_=csr_from_array(supplied_product,index_dtype)
    native_result=(t(supplied_product.shape,index_dtype),t([0,len(c_val)],index_dtype),
                   supplied.indptr,supplied.indices,supplied.values)
    calls=[]
    def native_call(primitive,*args):
        assert len(args)==10
        flags=dict(transpose_a=trans_a,transpose_b=trans_b,adjoint_a=adjoint_a,adjoint_b=adjoint_b)
        for key,flag in flags.items():assert primitive.attrs[key] is flag
        expected=(a_stored.shape,[0,len(a_val)],a_ptr,a_col,a_val,
                  b_stored.shape,[0,len(b_val)],b_ptr,b_col,b_val)
        for position,(actual,values) in enumerate(zip(args,expected)):
            tensor_fields(actual,values,value_dtype if position in (4,9) else index_dtype)
        calls.append(args);return native_result
    monkeypatch.setattr(sparse.SparseMatrixSparseMatMul,'__call__',native_call)
    result=sparse.csr_mm(a,b,trans_a=trans_a,trans_b=trans_b,adjoint_a=adjoint_a,adjoint_b=adjoint_b)
    assert len(calls)==1
    csr_fields(result,c_ptr,c_col,c_val,supplied_product.shape,index_dtype,value_dtype)


@pytest.mark.parametrize('value_dtype',[ms.float16,ms.float32])
def test_coalesce_real_cpu_duplicate_coordinate_sum(value_dtype):
    indices=t([[1,0,1,0],[0,1,0,1]],ms.int64)
    values=t([2.,3.,4.,-1.],value_dtype);shape=t([2,2],ms.int64)
    result=sparse.coalesce(indices,values,shape)
    assert isinstance(result,tuple) and len(result)==3
    tensor_fields(result[0],[[0,1],[1,0]],ms.int64)
    tensor_fields(result[1],[2.,6.],value_dtype)
    tensor_fields(result[2],[2,2],ms.int64)


@pytest.mark.parametrize('kind',['coo','row'])
def test_deprecated_sparse_construction_native_boundary(monkeypatch,capsys,kind):
    """Only the native Make* primitive call is supplied; returned storage is a genuine sparse object."""
    indices=t([[0,2],[1,1]],ms.int32) if kind=='coo' else t([0,1],ms.int32)
    values=t([2.,3.]) if kind=='coo' else t([[2.,3.,4.],[5.,6.,7.]])
    shape=(2,3)
    primitive=sparse.make_coo_tensor if kind=='coo' else sparse.make_row_tensor_inner
    primitive_type=type(primitive);original_call=primitive_type.__call__;calls=[];outputs=[]
    def native_call(registered,*args,**kwargs):
        if registered is not primitive:return original_call(registered,*args,**kwargs)
        assert not kwargs and len(args)==3
        assert args[0] is indices and args[1] is values and args[2] is shape
        if kind=='coo':output=ms.COOTensor(*args)
        else:
            from mindspore.common.sparse_tensor import RowTensorInner
            output=RowTensorInner(*args)
        calls.append(args);outputs.append(output);return output
    monkeypatch.setattr(primitive_type,'__call__',native_call)
    result=sparse.make_sparse_tensor(indices,values,shape) if kind=='coo' else sparse.make_row_tensor(indices,values,shape)
    assert result is outputs[0] and len(calls)==1
    tensor_fields(result.indices,[[0,2],[1,1]] if kind=='coo' else [0,1],ms.int32)
    tensor_fields(result.values,[2.,3.] if kind=='coo' else [[2.,3.,4.],[5.,6.,7.]],ms.float32)
    assert (result.shape if kind=='coo' else result.dense_shape)==shape
    assert 'deprecated' in capsys.readouterr().out


@pytest.mark.parametrize('shape,expected',[((2,),False),((),True),((1,1),True),((1,1,1),False)])
def test_sparse_scalar_metadata_uses_actual_tensor_shape(shape,expected):
    value=t(np.ones(shape,np.float32))
    assert sparse.is_scalar(value) is expected


@pytest.mark.parametrize('dtype',[ms.int32,ms.float64])
def test_sparse_typed_tensor_factory_returns_real_values(dtype):
    tensor_fields(sparse._make_tensor_with_dtype([2,3],dtype),[2,3],dtype)


@pytest.mark.parametrize('shape',[(2,3),(2,-1),ms.mutable([2,3],True)])
def test_sparse_shape_conversion_with_real_sequence_metadata(shape):
    assert sparse._convert_shape(shape)==tuple(shape)


@ms.jit
def convert_dynamic_shape(shape):return sparse._convert_shape(shape)


def test_sparse_shape_conversion_compiled_dynamic_length():
    assert convert_dynamic_shape(ms.mutable([2,3],True))==(-2,)


@pytest.mark.parametrize('operation',['csr_div','csr_mul'])
def test_invalid_dense_broadcast_native_failure_protocol(monkeypatch,operation):
    """Error propagation only: native validation is supplied, not claimed executed."""
    x=make_csr();dense=t(np.ones((1,1,1),np.float32));calls=[]
    assert x.ndim==2 and dense.ndim==3 and sparse.is_scalar(dense) is False
    rejection=ValueError('native boundary: dense shape cannot broadcast to sparse shape')
    primitive_class=sparse._csr_ops.CSRDiv if operation=='csr_div' else sparse._csr_ops.CSRMul
    def native_call(primitive,indptr,indices,values,shape,y):
        assert primitive.name==('CSRDiv' if operation=='csr_div' else 'CSRMul') and shape==(2,3)
        tensor_fields(indptr,[0,2,3],ms.int32);tensor_fields(indices,[0,2,1],ms.int32)
        tensor_fields(values,[1.,2.,3.],ms.float32);tensor_fields(y,np.ones((1,1,1)),ms.float32)
        calls.append((indptr,indices,values,shape,y));raise rejection
    monkeypatch.setattr(primitive_class,'__call__',native_call)
    with pytest.raises(ValueError,match='cannot broadcast') as error:getattr(sparse,operation)(x,dense)
    assert error.value is rejection and len(calls)==1


class DynamicMultilabelMargin(ms.nn.Cell):
    def construct(self,logits,target):return nn.multilabel_soft_margin_loss(logits,target)


@pytest.mark.parametrize('shape',[(2,2),(3,4)])
def test_multilabel_soft_margin_natural_dynamic_shape_cpu_graph(shape):
    network=DynamicMultilabelMargin()
    network.set_inputs(ms.Tensor(shape=[None,None],dtype=ms.float32),ms.Tensor(shape=[None,None],dtype=ms.float32))
    values=np.linspace(-1.,1.,np.prod(shape),dtype=np.float32).reshape(shape)
    target=(np.arange(np.prod(shape)).reshape(shape)%2).astype(np.float32)
    expected=(target*np.logaddexp(0,-values)+(1-target)*np.logaddexp(0,values)).mean()
    previous=ms.get_context('mode')
    try:
        ms.set_context(mode=ms.GRAPH_MODE)
        tensor_fields(network(t(values),t(target)),expected,ms.float32)
    finally:
        ms.set_context(mode=previous)


@ms.jit
def scalar_to_array_compiled(value):return array.scalar_to_array(value)


@pytest.mark.xfail(strict=True,raises=RuntimeError,reason='ScalarToArray has no matching graph evaluator in the installed MS 2.12 runtime')
@pytest.mark.parametrize('value',[3,2.5])
def test_deprecated_scalar_to_array_compiled_cpu(value):
    result=scalar_to_array_compiled(ms.mutable(value))
    assert isinstance(result,ms.Tensor) and result.shape==()
    np.testing.assert_allclose(result.asnumpy(),value)


@pytest.mark.parametrize('value,expected_dtype',[(3,ms.int64),(2.5,ms.float64)])
def test_scalar_to_array_native_primitive_protocol(monkeypatch,value,expected_dtype):
    """Protocol only: deprecated primitive is missing an evaluator; no native conversion is claimed."""
    supplied=t(value,expected_dtype);calls=[]
    def native_call(primitive,input_x):
        assert primitive.name=='ScalarToArray' and input_x is value
        calls.append(input_x);return supplied
    monkeypatch.setattr(array.P.ScalarToArray,'__call__',native_call)
    assert array.scalar_to_array(value) is supplied and calls==[value]
    tensor_fields(supplied,value,expected_dtype)
