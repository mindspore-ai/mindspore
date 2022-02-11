mindspore.ops
=============

可用于Cell的构造函数的算子。

.. code-block::

    import mindspore.ops as ops

MindSpore中 `mindspore.ops` 接口与上一版本相比，新增、删除和支持平台的变化信息请参考 `API Updates <https://gitee.com/mindspore/docs/blob/master/resource/api_updates/ops_api_updates.md>`_。

operations
----------

神经网络算子
^^^^^^^^^^^^

operations中的Primitive算子在使用前需要实例化。

.. cnmsplatformautosummary::
    :toctree: ops

    mindspore.ops.BiasAdd
    mindspore.ops.Elu
    mindspore.ops.GeLU
    mindspore.ops.L2Loss
    mindspore.ops.L2Normalize
    mindspore.ops.PReLU
    mindspore.ops.ReLUV2
    mindspore.ops.SeLU
    mindspore.ops.Sigmoid

Math类型算子
^^^^^^^^^^^^

.. cnmsplatformautosummary::
    :toctree: ops

    mindspore.ops.Add
    mindspore.ops.AddN
    mindspore.ops.Div
    mindspore.ops.Eps
    mindspore.ops.Erf
    mindspore.ops.Greater
    mindspore.ops.Inv
    mindspore.ops.LessEqual
    mindspore.ops.Log
    mindspore.ops.MatMul
    mindspore.ops.Mul
    mindspore.ops.Pow
    mindspore.ops.Sub

Array类型算子
^^^^^^^^^^^^^^

.. cnmsplatformautosummary::
    :toctree: ops

    mindspore.ops.Eye
    mindspore.ops.Fill
    mindspore.ops.Gather
    mindspore.ops.OnesLike
    mindspore.ops.Reshape
    mindspore.ops.Size
    mindspore.ops.Squeeze
    mindspore.ops.Tile
    mindspore.ops.ZerosLike


Random类型算子
^^^^^^^^^^^^^^^

.. cnmsplatformautosummary::
    :toctree: ops

    mindspore.ops.Gamma
    mindspore.ops.UniformReal


原语
----

.. cnmsplatformautosummary::
    :toctree: ops

    mindspore.ops.constexpr
    mindspore.ops.prim_attr_register
    mindspore.ops.Primitive
    mindspore.ops.PrimitiveWithCheck
    mindspore.ops.PrimitiveWithInfer


函数实现注册
--------------

.. cnmsplatformautosummary::
    :toctree: ops

    mindspore.ops.get_vm_impl_fn


算子信息注册
--------------

.. cnmsplatformautosummary::
    :toctree: ops

    mindspore.ops.DataType
