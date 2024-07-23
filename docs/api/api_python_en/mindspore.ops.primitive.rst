mindspore.ops.primitive
========================

operators that can be used for constructor function of Cell

For more information about dynamic shape support status, please refer to `Dynamic Shape Support Status of primitive Interface <https://mindspore.cn/docs/en/master/note/dynamic_shape_primitive.html>`_ .

For more information about the support for the bfloat16 data type, please refer to `Support List <https://mindspore.cn/docs/en/master/note/bfloat16_support.html>`_ .

For the details about the usage constraints of each operator in the operator parallel process,
refer to `Usage Constraints During Operator Parallel <https://www.mindspore.cn/docs/en/master/note/operator_list_parallel.html>`_ .

The module import method is as follows:

.. code-block::

    import mindspore.ops as ops

Compared with the previous version, the added, deleted and supported platforms change information of `mindspore.ops.primitive` operators in MindSpore, please refer to the link `mindspore.ops.primitive API Interface Change <https://gitee.com/mindspore/docs/blob/master/resource/api_updates/ops_api_updates_en.md>`_ .

Operator Primitives
-------------------

.. autosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.Primitive
    mindspore.ops.PrimitiveWithCheck
    mindspore.ops.PrimitiveWithInfer

Neural Network Layer Operators
------------------------------

Neural Network
^^^^^^^^^^^^^^

.. msplatwarnautosummary::
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
    mindspore.ops.EmbeddingLookup
    mindspore.ops.Padding
    mindspore.ops.ResizeBicubic
    mindspore.ops.ResizeNearestNeighbor

Loss Function
^^^^^^^^^^^^^

.. msplatwarnautosummary::
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

Activation Function
^^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
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

Optimizer
^^^^^^^^^

.. msplatwarnautosummary::
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

Distance Function
^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.Cdist
    mindspore.ops.EditDistance
    mindspore.ops.LpNorm
    mindspore.ops.Pdist

Sampling Operator
^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst
    
    mindspore.ops.ComputeAccidentalHits
    mindspore.ops.LogUniformCandidateSampler
    mindspore.ops.UniformCandidateSampler
    
Image Processing
^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
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
    mindspore.ops.UpsampleNearest3D
    mindspore.ops.UpsampleTrilinear3D

Text Processing
^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.NoRepeatNGram

Mathematical Operators
------------------------

.. msplatwarnautosummary::
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

Element-wise Operator
^^^^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
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
    mindspore.ops.Pow
    mindspore.ops.Polar
    mindspore.ops.Polygamma
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


Reduction Operator
^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
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

Comparison Operator
^^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
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

Linear Algebraic Operator
^^^^^^^^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
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

Tensor Operation Operator
--------------------------

Tensor Construction
^^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
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

Random Generation Operator
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
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

Array Operation
^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
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

Type Conversion
^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.ScalarCast
    mindspore.ops.ScalarToTensor
    mindspore.ops.TupleToArray
    
Parameter Operation Operator
----------------------------

.. msplatwarnautosummary::
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

Data Operation Operator
-----------------------

.. msplatwarnautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.GetNext

Communication Operator
----------------------

Distributed training involves communication operations for data transfer. For more details, refer to `Distributed Set Communication Primitives <https://www.mindspore.cn/docs/en/master/api_python/samples/ops/communicate_ops.html>`_ .

Note that the APIs in the following list need to preset communication environment variables. For Ascend/GPU/CPU devices,
it is recommended to use the msrun startup method without any third-party or configuration file dependencies.
Please see the `msrun start up \
<https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/msrun_launcher.html>`_ for more details.

.. msplatwarnautosummary::
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

Debugging Operator
------------------

.. msplatwarnautosummary::
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

Sparse Operator
---------------

.. msplatwarnautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.SparseTensorDenseMatmul
    mindspore.ops.SparseToDense

Frame Operators
---------------

.. msplatwarnautosummary::
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

Operator Information Registration
---------------------------------

.. autosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.AiCPURegOp
    mindspore.ops.CustomRegOp
    mindspore.ops.DataType
    mindspore.ops.TBERegOp
    mindspore.ops.get_vm_impl_fn

Customizing Operator
--------------------

.. msplatwarnautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.Custom
    mindspore.ops.custom_info_register
    mindspore.ops.kernel

Spectral Operator
-----------------

.. msplatwarnautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.BartlettWindow
    mindspore.ops.BlackmanWindow

