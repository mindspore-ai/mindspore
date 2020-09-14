/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef MINDSPORE_LITE_INTERNAL_INCLUDE_MODEL_H
#define MINDSPORE_LITE_INTERNAL_INCLUDE_MODEL_H
#include "internal/include/lite_utils.h"
#include "nnacl/op_base.h"

using PrimitiveC = OpParameter;
enum NodeType {
  NodeType_ValueNode = 0,
  NodeType_Parameter = 1,
  NodeType_CNode = 2,
  NodeType_MIN = NodeType_ValueNode,
  NodeType_MAX = NodeType_CNode
};

enum KernelType {
    Concat,
    SoftMax,
    Activation,
    Conv2D,
    FusedBatchNorm,
    BatchNorm,
    BiasAdd,
    Pooling,
    ROIPooling,
    DepthwiseConv2D,
    DeDepthwiseConv2D,
    Resize,
    DetectionPostProcess,
    FullConnection,
    Mean,
    DeConv2D,
    Scale,
    Reshape,
    Eltwise,
    NetOutput,
    Add,
    Sub,
    MatMul,
    StridedSlice,
    Power,
    Slice,
    Stack,
    Mul,
    RealDiv,
    Pad,
    Maximum,
    Minimum,
    PReLU,
    LeakyReLU,
    ArgMax,
    ArgMin,
    Exp,
    Crop,
    Range,
    Rsqrt,
    ExpandDims,
    Tile,
    Cast,
    Shape,
    Nchw2Nhwc,
    Nhwc2Nchw,
    QuantDTypeCast,
    Split,
    Permute,
    FakeQuantWithMinMaxVars,
    Equal,
    Less,
    Greater,
    NotEqual,
    LessEqual,
    GreaterEqual,
    Min,
    Floor,
    Abs,
    Neg,
    Cos,
    Sin,
    Sqrt,
    Square,
    Constant,
    Log,
    Tan,
    Atan,
    Asin,
    Clip,
    Transpose,
    Squeeze,
    Unsqueeze,
    Upsample,
    Dropout,
    Broadcast,
    BroadcastTo,
    Lrn,
    ZerosLike,
    TopK,
    SpaceToDepth,
    SpaceToBatch,
    SparseToDense,
    ReverseSequence,
    Rank,
    Gather,
    GatherNd,
    Fill,
    Elu,
    DepthToSpace,
    BatchToSpace,
    AddN,
    Ceil,
    EmbeddingLookup,
    EmbeddingLookupSparse,
    FloorDiv,
    FloorMod,
    L2Norm,
    LocalResponseNormalization,
    MatrixDiag,
    Reduce,
    Reverse,
    Round,
    Select,
    Scatter,
    ScatterND,
    ConstantOfShape,
    Unique,
    Unstack,
    LogicalAnd,
    LogicalOr,
    LogicalXor,
    LogicalNot,
    OnnxInt8Quantize,
    OnnxInt8Dequantize,
    FakeQuantWithMinMax,
    FakeQuantWithMinMaxPerChannel,
    BatchNormFold,
    MulFold,
    AddFold,
    SquaredDifference,
    Flatten,
    FlattenGrad,
    TupleGetItem,
    Div,
    Where,
    OneHot,
    Lstm,
    Conv2DGradFilter,
    Conv2DGradInput,
    PoolingGrad,
    BNGrad,
    BNGradInput,
    ApplyMomentum,
    BiasGrad,
    SoftmaxCrossEntropy,
    AddGrad,
    SubGrad,
    MulGrad,
    DivGrad,
    PowerGrad,
    ActivationGrad,
    PriorBox,
    SpaceToBatchND,
    Depend,
    Return,
    MakeTuple,
    ToFormat,
    Proposal,
    Custom,
    BlackBox,
    NegGrad,
    LogGrad,
    BatchToSpaceND,
    END,
};

enum ActivationType {
    NO_ACTIVATION = 0,
    RELU = 1,
    SIGMOID = 2,
    RELU6 = 3,
    ELU = 4,
    LEAKY_RELU = 5,
    ABS = 6,
    RELU1 = 7,
    SOFTSIGN = 8,
    SOFTPLUS = 9,
    TANH = 10,
    SELU = 11,
    HSWISH = 12,
    HSIGMOID = 13,
    THRESHOLDRELU = 14,
    LINEAR = 15,
    UNKNOW = 16
};

typedef struct Node {
  String name_;
  NodeType node_type_;
  PrimitiveC *primitive_;
  Uint32Vector input_indices_;
  Uint32Vector output_indices_;
} Node;

typedef struct Model {
  String name_;
  String version_;
  TensorPtrVector all_tensors_;
  Uint32Vector input_indices_;
  Uint32Vector output_indices_;
  NodePtrVector nodes_;
  char *buf;

  /// \brief Static method to create a Model pointer.
  ///
  /// \param[in] model_buf Define the buffer read from a model file.
  /// \param[in] size Define bytes number of model buffer.
  ///
  /// \return Pointer of MindSpore Lite Model.
  static Model *Import(const char *model_buf, size_t size);

  /// \brief Free all the temporary buffer
  void Free();
} Model;

#endif  // MINDSPORE_LITE_INTERNAL_INCLUDE_MODEL_H
