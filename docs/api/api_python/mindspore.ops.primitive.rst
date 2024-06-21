mindspore.ops.primitive
========================

可用于Cell的构造函数的算子。

动态shape的支持情况详见 `算子动态shape支持情况 <https://mindspore.cn/docs/zh-CN/master/note/dynamic_shape_primitive.html>`_ 。

bfloat16数据类型的支持情况详见 `支持列表 <https://mindspore.cn/docs/zh-CN/master/note/bfloat16_support.html>`_ 。

算子级并行过程各算子的使用约束详见 `算子级并行使用约束 <https://www.mindspore.cn/docs/zh-CN/master/note/operator_list_parallel.html>`_ 。

模块导入方法如下：

.. code-block::

    import mindspore.ops as ops

MindSpore中 `mindspore.ops.primitive` 接口与上一版本相比，新增、删除和支持平台的变化信息请参考 `mindspore.ops.primitive API接口变更 <https://gitee.com/mindspore/docs/blob/master/resource/api_updates/ops_api_updates_cn.md>`_ 。

算子原语
----------

.. mscnautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.Primitive
    mindspore.ops.PrimitiveWithCheck
    mindspore.ops.PrimitiveWithInfer

神经网络层算子
----------------

神经网络
^^^^^^^^^^

.. mscnplatwarnautosummary::
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
    mindspore.ops.Dense
    mindspore.ops.Dropout
    mindspore.ops.Dropout2D
    mindspore.ops.Dropout3D
    mindspore.ops.DynamicGRUV2
    mindspore.ops.DynamicRNN
    mindspore.ops.EmbeddingLookup
    mindspore.ops.Flatten
    mindspore.ops.FractionalMaxPool3DWithFixedKsize
    mindspore.ops.GridSampler2D
    mindspore.ops.GridSampler3D
    mindspore.ops.LayerNorm
    mindspore.ops.LRN
    mindspore.ops.LSTM
    mindspore.ops.MaxPool
    mindspore.ops.MaxPool3D
    mindspore.ops.MaxPool3DWithArgmax
    mindspore.ops.MaxUnpool2D
    mindspore.ops.MaxUnpool3D
    mindspore.ops.MirrorPad
    mindspore.ops.Pad
    mindspore.ops.Padding
    mindspore.ops.ResizeBicubic
    mindspore.ops.ResizeNearestNeighbor
    mindspore.ops.UpsampleNearest3D
    mindspore.ops.UpsampleTrilinear3D

损失函数
^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.BCEWithLogitsLoss
    mindspore.ops.BinaryCrossEntropy
    mindspore.ops.CTCLoss
    mindspore.ops.CTCLossV2
    mindspore.ops.KLDivLoss
    mindspore.ops.L2Loss
    mindspore.ops.MultilabelMarginLoss
    mindspore.ops.MultiMarginLoss
    mindspore.ops.NLLLoss
    mindspore.ops.RNNTLoss
    mindspore.ops.SigmoidCrossEntropyWithLogits
    mindspore.ops.SmoothL1Loss
    mindspore.ops.SoftMarginLoss
    mindspore.ops.SoftmaxCrossEntropyWithLogits
    mindspore.ops.SparseSoftmaxCrossEntropyWithLogits
    mindspore.ops.TripletMarginLoss

激活函数
^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.CeLU
    mindspore.ops.Elu
    mindspore.ops.FastGeLU
    mindspore.ops.GeLU
    mindspore.ops.GLU
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

.. mscnplatwarnautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.Adam
    mindspore.ops.AdamWeightDecay
    mindspore.ops.AdaptiveAvgPool2D
    mindspore.ops.AdaptiveAvgPool3D
    mindspore.ops.ApplyAdadelta
    mindspore.ops.ApplyAdagrad
    mindspore.ops.ApplyAdagradDA
    mindspore.ops.ApplyAdagradV2
    mindspore.ops.ApplyAdaMax
    mindspore.ops.ApplyAdamWithAmsgradV2
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
    mindspore.ops.SparseApplyAdagradV2
    mindspore.ops.SparseApplyProximalAdagrad
    mindspore.ops.SGD
    mindspore.ops.SparseApplyFtrl
    mindspore.ops.SparseApplyFtrlV2

距离函数
^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.Cdist
    mindspore.ops.EditDistance
    mindspore.ops.LpNorm
    mindspore.ops.Pdist

采样算子
^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst
    
    mindspore.ops.ComputeAccidentalHits
    mindspore.ops.LogUniformCandidateSampler
    mindspore.ops.UniformCandidateSampler
    
图像处理
^^^^^^^^^^

.. mscnplatwarnautosummary::
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
    mindspore.ops.ResizeBilinearV2
    mindspore.ops.ROIAlign

文本处理
^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.NoRepeatNGram

特征值检测
^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.silent_check.ASDBase

数学运算算子
------------

.. mscnplatwarnautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.Bincount
    mindspore.ops.Cholesky
    mindspore.ops.Complex
    mindspore.ops.ComplexAbs
    mindspore.ops.Cross
    mindspore.ops.FFTWithSize
    mindspore.ops.Gcd

逐元素运算
^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
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
    mindspore.ops.Angle
    mindspore.ops.Asin
    mindspore.ops.Asinh
    mindspore.ops.Atan
    mindspore.ops.Atan2
    mindspore.ops.Atanh
    mindspore.ops.BesselI0
    mindspore.ops.BesselI0e
    mindspore.ops.BesselI1
    mindspore.ops.BesselI1e
    mindspore.ops.BesselJ0
    mindspore.ops.BesselJ1
    mindspore.ops.BesselK0
    mindspore.ops.BesselK0e
    mindspore.ops.BesselK1
    mindspore.ops.BesselK1e
    mindspore.ops.BesselY0
    mindspore.ops.BesselY1
    mindspore.ops.BitwiseAnd
    mindspore.ops.BitwiseOr
    mindspore.ops.BitwiseXor
    mindspore.ops.Ceil
    mindspore.ops.Conj
    mindspore.ops.Cos
    mindspore.ops.Cosh
    mindspore.ops.Digamma
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
    mindspore.ops.Geqrf
    mindspore.ops.Imag
    mindspore.ops.Inv
    mindspore.ops.Invert
    mindspore.ops.Lerp
    mindspore.ops.Log
    mindspore.ops.Log1p
    mindspore.ops.LogicalAnd
    mindspore.ops.LogicalNot
    mindspore.ops.LogicalOr
    mindspore.ops.LogicalXor
    mindspore.ops.Logit
    mindspore.ops.Mod
    mindspore.ops.Mul
    mindspore.ops.MulNoNan
    mindspore.ops.Neg
    mindspore.ops.NextAfter
    mindspore.ops.Polar
    mindspore.ops.Polygamma
    mindspore.ops.Pow
    mindspore.ops.Real
    mindspore.ops.RealDiv
    mindspore.ops.Reciprocal
    mindspore.ops.Rint
    mindspore.ops.Round
    mindspore.ops.Rsqrt
    mindspore.ops.Sign
    mindspore.ops.Sin
    mindspore.ops.Sinc
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
    mindspore.ops.Zeta


Reduction算子
^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.Argmax
    mindspore.ops.ArgMaxWithValue
    mindspore.ops.Argmin
    mindspore.ops.ArgMinWithValue
    mindspore.ops.Median
    mindspore.ops.ReduceAll
    mindspore.ops.ReduceAny
    mindspore.ops.ReduceMax
    mindspore.ops.ReduceMean
    mindspore.ops.ReduceMin
    mindspore.ops.ReduceProd
    mindspore.ops.ReduceSum

比较算子
^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.ApproximateEqual
    mindspore.ops.Equal
    mindspore.ops.EqualCount
    mindspore.ops.Greater
    mindspore.ops.GreaterEqual
    mindspore.ops.InTopK
    mindspore.ops.IsFinite
    mindspore.ops.IsInf
    mindspore.ops.IsNan
    mindspore.ops.Less
    mindspore.ops.LessEqual
    mindspore.ops.Maximum
    mindspore.ops.Minimum
    mindspore.ops.NotEqual
    mindspore.ops.TopK

线性代数算子
^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.BatchMatMul
    mindspore.ops.BiasAdd
    mindspore.ops.Ger
    mindspore.ops.MatMul
    mindspore.ops.MatrixInverse
    mindspore.ops.Ormqr
    mindspore.ops.Orgqr
    mindspore.ops.Svd

Tensor操作算子
----------------

Tensor创建
^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
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

.. mscnplatwarnautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.Bernoulli
    mindspore.ops.Gamma
    mindspore.ops.Multinomial
    mindspore.ops.MultinomialWithReplacement
    mindspore.ops.RandomCategorical
    mindspore.ops.RandomChoiceWithMask
    mindspore.ops.RandomGamma
    mindspore.ops.RandomPoisson
    mindspore.ops.Randperm
    mindspore.ops.RandpermV2
    mindspore.ops.StandardLaplace
    mindspore.ops.StandardNormal
    mindspore.ops.UniformInt
    mindspore.ops.UniformReal

Array操作
^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.AffineGrid
    mindspore.ops.BatchToSpace
    mindspore.ops.BatchToSpaceND
    mindspore.ops.BroadcastTo
    mindspore.ops.Cast
    mindspore.ops.ChannelShuffle
    mindspore.ops.Col2Im
    mindspore.ops.Concat
    mindspore.ops.Cummax
    mindspore.ops.Cummin
    mindspore.ops.CumProd
    mindspore.ops.CumSum
    mindspore.ops.DataFormatDimMap
    mindspore.ops.DepthToSpace
    mindspore.ops.Diag
    mindspore.ops.DType
    mindspore.ops.ExpandDims
    mindspore.ops.FillDiagonal
    mindspore.ops.FillV2
    mindspore.ops.FloatStatus
    mindspore.ops.Fmax
    mindspore.ops.Gather
    mindspore.ops.GatherD
    mindspore.ops.GatherNd
    mindspore.ops.HammingWindow
    mindspore.ops.Heaviside
    mindspore.ops.HistogramFixedWidth
    mindspore.ops.Hypot
    mindspore.ops.Identity
    mindspore.ops.Igamma
    mindspore.ops.Igammac
    mindspore.ops.Im2Col
    mindspore.ops.IndexAdd
    mindspore.ops.IndexFill
    mindspore.ops.IndexPut
    mindspore.ops.InplaceAdd
    mindspore.ops.InplaceIndexAdd
    mindspore.ops.InplaceSub
    mindspore.ops.InplaceUpdate
    mindspore.ops.InplaceUpdateV2
    mindspore.ops.InvertPermutation
    mindspore.ops.IsClose
    mindspore.ops.Lcm
    mindspore.ops.LeftShift
    mindspore.ops.LogSpace
    mindspore.ops.LuUnpack
    mindspore.ops.MaskedFill
    mindspore.ops.MaskedScatter
    mindspore.ops.MaskedSelect
    mindspore.ops.MatrixBandPart
    mindspore.ops.MatrixDiagPartV3
    mindspore.ops.MatrixDiagV3
    mindspore.ops.MatrixSetDiagV3
    mindspore.ops.MatrixSolve
    mindspore.ops.Meshgrid
    mindspore.ops.Mvlgamma
    mindspore.ops.NanToNum
    mindspore.ops.NonZero
    mindspore.ops.ParallelConcat
    mindspore.ops.PopulationCount
    mindspore.ops.RandomShuffle
    mindspore.ops.Range
    mindspore.ops.Rank
    mindspore.ops.Renorm
    mindspore.ops.Reshape
    mindspore.ops.ReverseSequence
    mindspore.ops.ReverseV2
    mindspore.ops.RightShift
    mindspore.ops.ScatterNd
    mindspore.ops.ScatterNdDiv
    mindspore.ops.ScatterNdMax
    mindspore.ops.ScatterNdMin
    mindspore.ops.ScatterNdMul
    mindspore.ops.SearchSorted
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
    mindspore.ops.Trace
    mindspore.ops.Transpose
    mindspore.ops.Tril
    mindspore.ops.TrilIndices
    mindspore.ops.Triu
    mindspore.ops.TriuIndices
    mindspore.ops.Unique
    mindspore.ops.UniqueConsecutive
    mindspore.ops.UniqueWithPad
    mindspore.ops.UnsortedSegmentMax
    mindspore.ops.UnsortedSegmentMin
    mindspore.ops.UnsortedSegmentProd
    mindspore.ops.UnsortedSegmentSum
    mindspore.ops.Unstack

类型转换
^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.ScalarCast
    mindspore.ops.ScalarToTensor
    mindspore.ops.TupleToArray
    
Parameter操作算子
--------------------

.. mscnplatwarnautosummary::
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

.. mscnplatwarnautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.GetNext

通信算子
----------------

在分布式训练中进行数据传输涉及通信操作，详情请参考 `分布式集合通信原语 <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/ops/communicate_ops.html>`_ 。

注意，以下列表中的接口需要先配置好通信环境变量。

针对Ascend/GPU/CPU设备，推荐使用msrun启动方式，无第三方以及配置文件依赖。详见 `msrun启动 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/msrun_launcher.html>`_ 。

.. mscnplatwarnautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.AllGather
    mindspore.ops.AllReduce
    mindspore.ops.AlltoAll
    mindspore.ops.Barrier
    mindspore.ops.Broadcast
    mindspore.ops.CollectiveGather
    mindspore.ops.CollectiveScatter
    mindspore.ops.NeighborExchangeV2
    mindspore.ops.Receive
    mindspore.ops.ReduceOp
    mindspore.ops.ReduceScatter
    mindspore.ops.Reduce
    mindspore.ops.Send

调试算子
----------------

.. mscnplatwarnautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.HistogramSummary
    mindspore.ops.ImageSummary
    mindspore.ops.ScalarSummary
    mindspore.ops.TensorSummary
    mindspore.ops.TensorDump
    mindspore.ops.Print
    mindspore.ops.NPUAllocFloatStatus
    mindspore.ops.NPUClearFloatStatus
    mindspore.ops.NPUGetFloatStatus

稀疏算子
----------------

.. mscnplatwarnautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.SparseTensorDenseMatmul
    mindspore.ops.SparseToDense

框架算子
----------------

.. mscnplatwarnautosummary::
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

.. mscnplatwarnautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.Custom
    mindspore.ops.custom_info_register
    mindspore.ops.kernel

光谱算子
----------

.. mscnplatwarnautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.BartlettWindow
    mindspore.ops.BlackmanWindow

