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

enum KernelType : int {
    KernelType_Concat = 0,
    KernelType_SoftMax,
    KernelType_Activation,
    KernelType_Conv2D,
    KernelType_FusedBatchNorm,
    KernelType_BatchNorm,
    KernelType_BiasAdd,
    KernelType_Pooling,
    KernelType_ROIPooling,
    KernelType_DepthwiseConv2D,
    KernelType_DeDepthwiseConv2D,
    KernelType_Resize,
    KernelType_DetectionPostProcess,
    KernelType_FullConnection,
    KernelType_Mean,
    KernelType_DeConv2D,
    KernelType_Scale,
    KernelType_Reshape,
    KernelType_Eltwise,
    KernelType_NetOutput,
    KernelType_Add,
    KernelType_Sub,
    KernelType_MatMul,
    KernelType_StridedSlice,
    KernelType_Power,
    KernelType_Slice,
    KernelType_Stack,
    KernelType_Mul,
    KernelType_RealDiv,
    KernelType_Pad,
    KernelType_Maximum,
    KernelType_Minimum,
    KernelType_PReLU,
    KernelType_LeakyReLU,
    KernelType_ArgMax,
    KernelType_ArgMin,
    KernelType_Exp,
    KernelType_Crop,
    KernelType_Range,
    KernelType_Rsqrt,
    KernelType_ExpandDims,
    KernelType_Tile,
    KernelType_Cast,
    KernelType_Shape,
    KernelType_Nchw2Nhwc,
    KernelType_Nhwc2Nchw,
    KernelType_QuantDTypeCast,
    KernelType_Split,
    KernelType_Permute,
    KernelType_FakeQuantWithMinMaxVars,
    KernelType_Equal,
    KernelType_Less,
    KernelType_Greater,
    KernelType_NotEqual,
    KernelType_LessEqual,
    KernelType_GreaterEqual,
    KernelType_Min,
    KernelType_Floor,
    KernelType_Abs,
    KernelType_Neg,
    KernelType_Cos,
    KernelType_Sin,
    KernelType_Sqrt,
    KernelType_Square,
    KernelType_Constant,
    KernelType_Log,
    KernelType_Tan,
    KernelType_Atan,
    KernelType_Asin,
    KernelType_Clip,
    KernelType_Transpose,
    KernelType_Squeeze,
    KernelType_Unsqueeze,
    KernelType_Upsample,
    KernelType_Dropout,
    KernelType_Broadcast,
    KernelType_BroadcastTo,
    KernelType_Lrn,
    KernelType_ZerosLike,
    KernelType_TopK,
    KernelType_SpaceToDepth,
    KernelType_SpaceToBatch,
    KernelType_SparseToDense,
    KernelType_ReverseSequence,
    KernelType_Rank,
    KernelType_Gather,
    KernelType_GatherNd,
    KernelType_Fill,
    KernelType_Elu,
    KernelType_DepthToSpace,
    KernelType_BatchToSpace,
    KernelType_AddN,
    KernelType_Ceil,
    KernelType_EmbeddingLookup,
    KernelType_EmbeddingLookupSparse,
    KernelType_FloorDiv,
    KernelType_FloorMod,
    KernelType_L2Norm,
    KernelType_LocalResponseNormalization,
    KernelType_MatrixDiag,
    KernelType_Reduce,
    KernelType_Reverse,
    KernelType_Round,
    KernelType_Select,
    KernelType_Scatter,
    KernelType_ScatterND,
    KernelType_ConstantOfShape,
    KernelType_Unique,
    KernelType_Unstack,
    KernelType_LogicalAnd,
    KernelType_LogicalOr,
    KernelType_LogicalXor,
    KernelType_LogicalNot,
    KernelType_OnnxInt8Quantize,
    KernelType_OnnxInt8Dequantize,
    KernelType_FakeQuantWithMinMax,
    KernelType_FakeQuantWithMinMaxPerChannel,
    KernelType_BatchNormFold,
    KernelType_MulFold,
    KernelType_AddFold,
    KernelType_SquaredDifference,
    KernelType_Flatten,
    KernelType_FlattenGrad,
    KernelType_TupleGetItem,
    KernelType_Div,
    KernelType_Where,
    KernelType_OneHot,
    KernelType_Lstm,
    KernelType_Conv2DGradFilter,
    KernelType_Conv2DGradInput,
    KernelType_PoolingGrad,
    KernelType_BNGrad,
    KernelType_BNGradInput,
    KernelType_ApplyMomentum,
    KernelType_BiasGrad,
    KernelType_SoftmaxCrossEntropy,
    KernelType_AddGrad,
    KernelType_SubGrad,
    KernelType_MulGrad,
    KernelType_DivGrad,
    KernelType_PowerGrad,
    KernelType_ActivationGrad,
    KernelType_PriorBox,
    KernelType_SpaceToBatchND,
    KernelType_Depend,
    KernelType_Return,
    KernelType_MakeTuple,
    KernelType_ToFormat,
    KernelType_Proposal,
    KernelType_Custom,
    KernelType_BlackBox,
    KernelType_NegGrad,
    KernelType_LogGrad,
    KernelType_BatchToSpaceND,
    KernelType_END,
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

enum ReduceMode {
  ReduceMode_ReduceMean = 0,
  ReduceMode_ReduceMax = 1,
  ReduceMode_ReduceMin = 2,
  ReduceMode_ReduceProd = 3,
  ReduceMode_ReduceSum = 4,
  ReduceMode_ReduceSumSquare = 5,
  ReduceMode_ReduceASum = 6,
  ReduceMode_MIN = ReduceMode_ReduceMean,
  ReduceMode_MAX = ReduceMode_ReduceASum
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
