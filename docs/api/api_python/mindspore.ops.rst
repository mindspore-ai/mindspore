mindspore.ops
=============

可用于Cell的构造函数的算子。

.. code-block::

    import mindspore.ops as ops

MindSpore中 `mindspore.ops` 接口与上一版本相比，新增、删除和支持平台的变化信息请参考 `mindspore.ops API接口变更 <https://gitee.com/mindspore/docs/blob/r1.10/resource/api_updates/ops_api_updates_cn.md>`_ 。

算子原语
----------

.. mscnautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.Primitive
    mindspore.ops.PrimitiveWithCheck
    mindspore.ops.PrimitiveWithInfer

装饰器
--------

.. mscnautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.constexpr
    mindspore.ops.custom_info_register
    mindspore.ops.ms_kernel
    mindspore.ops.op_info_register
    mindspore.ops.prim_attr_register

神经网络层算子
----------------

神经网络
^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.AvgPool
    mindspore.ops.AvgPool3D
    mindspore.ops.BatchNorm
    mindspore.ops.Conv2D
    mindspore.ops.Conv2DTranspose
    mindspore.ops.Conv3D
    mindspore.ops.Conv3DTranspose
    mindspore.ops.CTCGreedyDecoder
    mindspore.ops.Dropout
    mindspore.ops.Dropout2D
    mindspore.ops.Dropout3D
    mindspore.ops.DynamicGRUV2
    mindspore.ops.DynamicRNN
    mindspore.ops.Flatten
    mindspore.ops.LayerNorm
    mindspore.ops.LRN
    mindspore.ops.LSTM
    mindspore.ops.MaxPool
    mindspore.ops.MaxPool3D
    mindspore.ops.MaxPoolWithArgmax
    mindspore.ops.MirrorPad
    mindspore.ops.Pad
    mindspore.ops.EmbeddingLookup
    mindspore.ops.Padding
    mindspore.ops.ResizeNearestNeighbor
    mindspore.ops.ResizeBilinear

损失函数
^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.BCEWithLogitsLoss
    mindspore.ops.BinaryCrossEntropy
    mindspore.ops.CTCLoss
    mindspore.ops.KLDivLoss
    mindspore.ops.L2Loss
    mindspore.ops.NLLLoss
    mindspore.ops.RNNTLoss
    mindspore.ops.SigmoidCrossEntropyWithLogits
    mindspore.ops.SmoothL1Loss
    mindspore.ops.SoftMarginLoss
    mindspore.ops.SoftmaxCrossEntropyWithLogits
    mindspore.ops.SparseSoftmaxCrossEntropyWithLogits

激活函数
^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.Elu
    mindspore.ops.FastGeLU
    mindspore.ops.GeLU
    mindspore.ops.HShrink
    mindspore.ops.HSigmoid
    mindspore.ops.HSwish
    mindspore.ops.LogSoftmax
    mindspore.ops.Mish
    mindspore.ops.PReLU
    mindspore.ops.ReLU
    mindspore.ops.ReLU6
    mindspore.ops.SeLU
    mindspore.ops.Sigmoid
    mindspore.ops.Softmax
    mindspore.ops.Softplus
    mindspore.ops.SoftShrink
    mindspore.ops.Softsign
    mindspore.ops.Tanh

优化器
^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.Adam
    mindspore.ops.AdamWeightDecay
    mindspore.ops.AdaptiveAvgPool2D
    mindspore.ops.ApplyAdadelta
    mindspore.ops.ApplyAdagrad
    mindspore.ops.ApplyAdagradDA
    mindspore.ops.ApplyAdagradV2
    mindspore.ops.ApplyAdaMax
    mindspore.ops.ApplyAddSign
    mindspore.ops.ApplyCenteredRMSProp
    mindspore.ops.ApplyFtrl
    mindspore.ops.ApplyGradientDescent
    mindspore.ops.ApplyMomentum
    mindspore.ops.ApplyPowerSign
    mindspore.ops.ApplyProximalAdagrad
    mindspore.ops.ApplyProximalGradientDescent
    mindspore.ops.ApplyRMSProp
    mindspore.ops.LARSUpdate
    mindspore.ops.SparseApplyAdagrad
    mindspore.ops.SparseApplyAdagradV2
    mindspore.ops.SparseApplyProximalAdagrad
    mindspore.ops.SGD
    mindspore.ops.SparseApplyFtrl
    mindspore.ops.SparseApplyFtrlV2

距离函数
^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.Cdist
    mindspore.ops.EditDistance
    mindspore.ops.LpNorm
    
采样算子
^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst
    
    mindspore.ops.ComputeAccidentalHits
    mindspore.ops.LogUniformCandidateSampler
    mindspore.ops.UniformCandidateSampler
    
图像处理
^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst
    
    mindspore.ops.BoundingBoxDecode
    mindspore.ops.BoundingBoxEncode
    mindspore.ops.CheckValid
    mindspore.ops.CropAndResize
    mindspore.ops.ExtractVolumePatches
    mindspore.ops.IOU
    mindspore.ops.L2Normalize
    mindspore.ops.NMSWithMask
    mindspore.ops.ROIAlign
    
文本处理
^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst
    
    mindspore.ops.NoRepeatNGram

数学运算算子
----------------

逐元素运算
^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.Abs
    mindspore.ops.AccumulateNV2
    mindspore.ops.ACos
    mindspore.ops.Acosh
    mindspore.ops.Add
    mindspore.ops.Addcdiv
    mindspore.ops.Addcmul
    mindspore.ops.AddN
    mindspore.ops.Asin
    mindspore.ops.Asinh
    mindspore.ops.Atan
    mindspore.ops.Atan2
    mindspore.ops.Atanh
    mindspore.ops.BesselI0e
    mindspore.ops.BesselI1e
    mindspore.ops.BitwiseAnd
    mindspore.ops.BitwiseOr
    mindspore.ops.BitwiseXor
    mindspore.ops.Ceil
    mindspore.ops.Conj
    mindspore.ops.Cos
    mindspore.ops.Cosh
    mindspore.ops.Div
    mindspore.ops.DivNoNan
    mindspore.ops.Einsum
    mindspore.ops.Erf
    mindspore.ops.Erfc
    mindspore.ops.Erfinv
    mindspore.ops.Exp
    mindspore.ops.Expm1
    mindspore.ops.Floor
    mindspore.ops.FloorDiv
    mindspore.ops.FloorMod
    mindspore.ops.Imag
    mindspore.ops.Inv
    mindspore.ops.Invert
    mindspore.ops.Lerp
    mindspore.ops.Log
    mindspore.ops.Log1p
    mindspore.ops.LogicalAnd
    mindspore.ops.LogicalNot
    mindspore.ops.LogicalOr
    mindspore.ops.Mod
    mindspore.ops.Mul
    mindspore.ops.MulNoNan
    mindspore.ops.Neg
    mindspore.ops.Pow
    mindspore.ops.Real
    mindspore.ops.RealDiv
    mindspore.ops.Reciprocal
    mindspore.ops.Rint
    mindspore.ops.Round
    mindspore.ops.Rsqrt
    mindspore.ops.Sign
    mindspore.ops.Sin
    mindspore.ops.Sinh
    mindspore.ops.Sqrt
    mindspore.ops.Square
    mindspore.ops.SquaredDifference
    mindspore.ops.SquareSumAll
    mindspore.ops.Sub
    mindspore.ops.Tan
    mindspore.ops.Trunc
    mindspore.ops.TruncateDiv
    mindspore.ops.TruncateMod
    mindspore.ops.Xdivy
    mindspore.ops.Xlogy
    

Reduction算子
^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.Argmax
    mindspore.ops.ArgMaxWithValue
    mindspore.ops.Argmin
    mindspore.ops.ArgMinWithValue
    mindspore.ops.ReduceAll
    mindspore.ops.ReduceAny
    mindspore.ops.ReduceMax
    mindspore.ops.ReduceMean
    mindspore.ops.ReduceMin
    mindspore.ops.ReduceProd
    mindspore.ops.ReduceSum

比较算子
^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.ApproximateEqual
    mindspore.ops.CheckBprop
    mindspore.ops.Equal
    mindspore.ops.EqualCount
    mindspore.ops.Greater
    mindspore.ops.GreaterEqual
    mindspore.ops.InTopK
    mindspore.ops.IsFinite
    mindspore.ops.IsInf
    mindspore.ops.IsInstance
    mindspore.ops.IsNan
    mindspore.ops.IsSubClass
    mindspore.ops.Less
    mindspore.ops.LessEqual
    mindspore.ops.Maximum
    mindspore.ops.Minimum
    mindspore.ops.NotEqual
    mindspore.ops.SameTypeShape
    mindspore.ops.TopK

线性代数算子
^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.BatchMatMul
    mindspore.ops.BiasAdd
    mindspore.ops.Ger
    mindspore.ops.MatMul
    mindspore.ops.MatrixInverse

Tensor操作算子
----------------

Tensor创建
^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.Eps
    mindspore.ops.Eye
    mindspore.ops.Fill
    mindspore.ops.LinSpace
    mindspore.ops.OneHot
    mindspore.ops.Ones
    mindspore.ops.OnesLike
    mindspore.ops.Zeros
    mindspore.ops.ZerosLike

随机生成算子
^^^^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.Gamma
    mindspore.ops.Multinomial
    mindspore.ops.RandomCategorical
    mindspore.ops.RandomChoiceWithMask
    mindspore.ops.Randperm
    mindspore.ops.StandardLaplace
    mindspore.ops.StandardNormal
    mindspore.ops.UniformInt
    mindspore.ops.UniformReal

Array操作
^^^^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.BatchToSpace
    mindspore.ops.BatchToSpaceND
    mindspore.ops.BroadcastTo
    mindspore.ops.Cast
    mindspore.ops.Concat
    mindspore.ops.CumProd
    mindspore.ops.CumSum
    mindspore.ops.DataFormatDimMap
    mindspore.ops.DepthToSpace
    mindspore.ops.DType
    mindspore.ops.ExpandDims
    mindspore.ops.FloatStatus
    mindspore.ops.Gather
    mindspore.ops.GatherD
    mindspore.ops.GatherNd
    mindspore.ops.HistogramFixedWidth
    mindspore.ops.Identity
    mindspore.ops.IndexAdd
    mindspore.ops.InplaceAdd
    mindspore.ops.InplaceSub
    mindspore.ops.InplaceUpdate
    mindspore.ops.InvertPermutation
    mindspore.ops.MaskedFill
    mindspore.ops.MaskedSelect
    mindspore.ops.Meshgrid
    mindspore.ops.ParallelConcat
    mindspore.ops.PopulationCount
    mindspore.ops.Range
    mindspore.ops.Rank
    mindspore.ops.Reshape
    mindspore.ops.ReverseSequence
    mindspore.ops.ReverseV2
    mindspore.ops.ScatterNd
    mindspore.ops.Select
    mindspore.ops.Shape
    mindspore.ops.Size
    mindspore.ops.Slice
    mindspore.ops.Sort
    mindspore.ops.SpaceToBatchND
    mindspore.ops.SpaceToDepth
    mindspore.ops.SparseGatherV2
    mindspore.ops.Split
    mindspore.ops.Squeeze
    mindspore.ops.Stack
    mindspore.ops.StridedSlice
    mindspore.ops.TensorScatterAdd
    mindspore.ops.TensorScatterDiv
    mindspore.ops.TensorScatterMax
    mindspore.ops.TensorScatterMin
    mindspore.ops.TensorScatterMul
    mindspore.ops.TensorScatterSub
    mindspore.ops.TensorScatterUpdate
    mindspore.ops.TensorShape
    mindspore.ops.Tile
    mindspore.ops.Transpose
    mindspore.ops.Unique
    mindspore.ops.UniqueWithPad
    mindspore.ops.UnsortedSegmentMax
    mindspore.ops.UnsortedSegmentMin
    mindspore.ops.UnsortedSegmentProd
    mindspore.ops.UnsortedSegmentSum
    mindspore.ops.Unstack

类型转换
^^^^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.ScalarCast
    mindspore.ops.ScalarToArray
    mindspore.ops.ScalarToTensor
    mindspore.ops.TupleToArray
    
Parameter操作算子
--------------------

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.Assign
    mindspore.ops.AssignAdd
    mindspore.ops.AssignSub
    mindspore.ops.ScatterAdd
    mindspore.ops.ScatterDiv
    mindspore.ops.ScatterMax
    mindspore.ops.ScatterMin
    mindspore.ops.ScatterMul
    mindspore.ops.ScatterNdAdd
    mindspore.ops.ScatterNdSub
    mindspore.ops.ScatterNdUpdate
    mindspore.ops.ScatterNonAliasingAdd
    mindspore.ops.ScatterSub
    mindspore.ops.ScatterUpdate

数据操作算子
----------------

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.GetNext

通信算子
----------------

注意，以下列表中的接口需要先配置好通信环境变量。

针对Ascend设备，用户需要准备rank表，设置rank_id和device_id，详见 `Ascend指导文档 \
<https://www.mindspore.cn/tutorials/experts/zh-CN/r1.10/parallel/train_ascend.html#准备环节>`_ 。

针对GPU设备，用户需要准备host文件和mpi，详见 `GPU指导文档 \
<https://www.mindspore.cn/tutorials/experts/zh-CN/r1.10/parallel/train_gpu.html#准备环节>`_ 。

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.AllGather
    mindspore.ops.AllReduce
    mindspore.ops.AlltoAll
    mindspore.ops.Broadcast
    mindspore.ops.NeighborExchange
    mindspore.ops.NeighborExchangeV2
    mindspore.ops.ReduceOp
    mindspore.ops.ReduceScatter

调试算子
----------------

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.HistogramSummary
    mindspore.ops.ImageSummary
    mindspore.ops.ScalarSummary
    mindspore.ops.TensorSummary
    mindspore.ops.Print
    mindspore.ops.NPUAllocFloatStatus
    mindspore.ops.NPUClearFloatStatus
    mindspore.ops.NPUGetFloatStatus

稀疏算子
----------------

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.SparseTensorDenseMatmul
    mindspore.ops.SparseToDense

框架算子
----------------

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.Depend
    mindspore.ops.GradOperation
    mindspore.ops.HookBackward
    mindspore.ops.HyperMap
    mindspore.ops.InsertGradientOf
    mindspore.ops.Map
    mindspore.ops.MultitypeFuncGraph
    mindspore.ops.Partial

算子信息注册
-------------

.. mscnautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.AiCPURegOp
    mindspore.ops.CustomRegOp
    mindspore.ops.DataType
    mindspore.ops.TBERegOp
    mindspore.ops.get_vm_impl_fn

自定义算子
-------------

.. mscnautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.Custom

