/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPERATOR_OPS_H_
#define MINDSPORE_CORE_OPERATOR_OPS_H_

#include <iostream>
#include <string>
#include <memory>
#include "ir/anf.h"
#include "ir/primitive.h"

namespace mindspore {
namespace prim {
// Here list all primitives used in backend or some special primitives used by core.
// Arithmetic
inline const PrimitivePtr kPrimScalarAdd = std::make_shared<Primitive>("scalar_add");
inline const PrimitivePtr kPrimScalarSub = std::make_shared<Primitive>("scalar_sub");
inline const PrimitivePtr kPrimScalarMul = std::make_shared<Primitive>("scalar_mul");
inline const PrimitivePtr kPrimScalarDiv = std::make_shared<Primitive>("scalar_div");
inline const PrimitivePtr kPrimScalarFloordiv = std::make_shared<Primitive>("scalar_floordiv");
inline const PrimitivePtr kPrimScalarMod = std::make_shared<Primitive>("scalar_mod");
inline const PrimitivePtr kPrimScalarPow = std::make_shared<Primitive>("scalar_pow");
inline const PrimitivePtr kPrimScalarTrunc = std::make_shared<Primitive>("scalar_trunc");
inline const PrimitivePtr kPrimScalarFloor = std::make_shared<Primitive>("scalar_floor");
inline const PrimitivePtr kPrimScalarUadd = std::make_shared<Primitive>("scalar_uadd");
inline const PrimitivePtr kPrimScalarUsub = std::make_shared<Primitive>("scalar_usub");
inline const PrimitivePtr kPrimScalarExp = std::make_shared<Primitive>("scalar_exp");
inline const PrimitivePtr kPrimScalarLog = std::make_shared<Primitive>("scalar_log");
inline const PrimitivePtr kPrimScalarSin = std::make_shared<Primitive>("scalar_sin");
inline const PrimitivePtr kPrimScalarCos = std::make_shared<Primitive>("scalar_cos");
inline const PrimitivePtr kPrimScalarTan = std::make_shared<Primitive>("scalar_tan");

// Comparisons
inline const PrimitivePtr kPrimScalarEq = std::make_shared<Primitive>("scalar_eq");
inline const PrimitivePtr kPrimScalarLt = std::make_shared<Primitive>("scalar_lt");
inline const PrimitivePtr kPrimScalarGt = std::make_shared<Primitive>("scalar_gt");
inline const PrimitivePtr kPrimScalarNe = std::make_shared<Primitive>("scalar_ne");
inline const PrimitivePtr kPrimScalarLe = std::make_shared<Primitive>("scalar_le");
inline const PrimitivePtr kPrimScalarGe = std::make_shared<Primitive>("scalar_ge");
inline const PrimitivePtr kPrimBoolNot = std::make_shared<Primitive>("bool_not");
inline const PrimitivePtr kPrimBoolAnd = std::make_shared<Primitive>("bool_and");
inline const PrimitivePtr kPrimBoolOr = std::make_shared<Primitive>("bool_or");
inline const PrimitivePtr kPrimBoolEq = std::make_shared<Primitive>("bool_eq");
inline const PrimitivePtr kPrimGreater = std::make_shared<Primitive>("Greater");
inline const PrimitivePtr kPrimGreaterEqual = std::make_shared<Primitive>("GreaterEqual");
inline const PrimitivePtr kPrimLess = std::make_shared<Primitive>("Less");
inline const PrimitivePtr kPrimLessEqual = std::make_shared<Primitive>("LessEqual");
inline const PrimitivePtr kPrimEqual = std::make_shared<Primitive>("Equal");
inline const PrimitivePtr kPrimNotEqual = std::make_shared<Primitive>("NotEqual");

inline const PrimitivePtr kPrimDistribute = std::make_shared<Primitive>("distribute");
inline const PrimitivePtr kPrimDot = std::make_shared<Primitive>("dot");
inline const PrimitivePtr kPrimIm2Col = std::make_shared<Primitive>("im2col");
inline const PrimitivePtr kPrimCol2Im = std::make_shared<Primitive>("col2im");
inline const PrimitivePtr kPrimIm2ColV1 = std::make_shared<Primitive>("im2col_v1");
inline const PrimitivePtr kPrimCol2ImV1 = std::make_shared<Primitive>("col2im_v1");

inline const PrimitivePtr kPrimLabelGoto = std::make_shared<Primitive>("LabelGoto");
inline const PrimitivePtr kPrimLabelSwitch = std::make_shared<Primitive>("LabelSwitch");
inline const PrimitivePtr kPrimLabelSet = std::make_shared<Primitive>("LabelSet");

// Arrays
inline const PrimitivePtr kPrimScalarToArray = std::make_shared<Primitive>("scalar_to_array");
inline const PrimitivePtr kPrimArrayToScalar = std::make_shared<Primitive>("array_to_scalar");
inline const PrimitivePtr kPrimBroadcastShape = std::make_shared<Primitive>("broadcast_shape");
inline const PrimitivePtr kPrimArrayMap = std::make_shared<Primitive>("array_map");
inline const PrimitivePtr kPrimArrayReduce = std::make_shared<Primitive>("array_reduce");
inline const PrimitivePtr kPrimCast = std::make_shared<Primitive>("Cast");
inline const PrimitivePtr kPrimConcat = std::make_shared<Primitive>("Concat");
inline const PrimitivePtr kPrimSqueeze = std::make_shared<Primitive>("Squeeze");
inline const PrimitivePtr kPrimTranspose = std::make_shared<Primitive>("Transpose");
inline const PrimitivePtr kPrimGatherV2 = std::make_shared<Primitive>("GatherV2");
inline const PrimitivePtr kPrimEmbeddingLookup = std::make_shared<Primitive>("EmbeddingLookup");
inline const PrimitivePtr kPrimEmbeddingLookupCommGrad = std::make_shared<Primitive>("EmbeddingLookupCommGrad");
inline const PrimitivePtr kPrimSize = std::make_shared<Primitive>("Size");
inline const PrimitivePtr kPrimArgMax = std::make_shared<Primitive>("Argmax");
inline const PrimitivePtr kPrimPack = std::make_shared<Primitive>("Pack");
inline const PrimitivePtr kPrimUnsortedSegmentSum = std::make_shared<Primitive>("UnsortedSegmentSum");
inline const PrimitivePtr kPrimUnsortedSegmentMin = std::make_shared<Primitive>("UnsortedSegmentMin");
inline const PrimitivePtr kPrimConcatOffset = std::make_shared<Primitive>("ConcatOffset");
inline const PrimitivePtr kPrimReshape = std::make_shared<Primitive>("Reshape");
inline const PrimitivePtr kPrimTile = std::make_shared<Primitive>("Tile");
inline const PrimitivePtr kPrimAddN = std::make_shared<Primitive>("AddN");
inline const PrimitivePtr KPrimTransData = std::make_shared<Primitive>("TransData");
inline const PrimitivePtr kPrimNMSWithMask = std::make_shared<Primitive>("NMSWithMask");
inline const PrimitivePtr kPrimPad = std::make_shared<Primitive>("Pad");
inline const PrimitivePtr kPrimArgMaxWithValue = std::make_shared<Primitive>("ArgMaxWithValue");
inline const PrimitivePtr kPrimUnique = std::make_shared<Primitive>("Unique");
inline const PrimitivePtr kPrimUniqueGrad = std::make_shared<Primitive>("UniqueGrad");

// NN
inline const PrimitivePtr kPrimFlatten = std::make_shared<Primitive>("Flatten");
inline const PrimitivePtr kPrimSoftmax = std::make_shared<Primitive>("Softmax");
inline const PrimitivePtr kPrimLogSoftmax = std::make_shared<Primitive>("LogSoftmax");
inline const PrimitivePtr kPrimLogSoftmaxGrad = std::make_shared<Primitive>("LogSoftmaxGrad");
inline const PrimitivePtr kPrimTanh = std::make_shared<Primitive>("Tanh");
inline const PrimitivePtr kPrimTanhGrad = std::make_shared<Primitive>("TanhGrad");
inline const PrimitivePtr kPrimPooling = std::make_shared<Primitive>("Pooling");
inline const PrimitivePtr kPrimPoolingGrad = std::make_shared<Primitive>("PoolingGrad");
inline const PrimitivePtr kPrimMaxPool = std::make_shared<Primitive>("MaxPool");
inline const PrimitivePtr kPrimMaxPoolGrad = std::make_shared<Primitive>("MaxPoolGrad");
inline const PrimitivePtr kPrimApplyCenteredRMSProp = std::make_shared<Primitive>("ApplyCenteredRMSProp");
inline const PrimitivePtr kPrimAvgPoolGrad = std::make_shared<Primitive>("AvgPoolGrad");
inline const PrimitivePtr kPrimAvgPoolGradVm = std::make_shared<Primitive>("AvgPoolGradVm");
inline const PrimitivePtr kPrimFusedBatchNorm = std::make_shared<Primitive>("FusedBatchNorm");
inline const PrimitivePtr kPrimConv2D = std::make_shared<Primitive>("Conv2D");
inline const PrimitivePtr kPrimFusedBatchNormGrad = std::make_shared<Primitive>("FusedBatchNormGrad");
inline const PrimitivePtr kPrimBatchNorm = std::make_shared<Primitive>("BatchNorm");
inline const PrimitivePtr kPrimBatchNormGrad = std::make_shared<Primitive>("BatchNormGrad");
inline const PrimitivePtr kPrimReluGrad = std::make_shared<Primitive>("ReluGrad");
inline const PrimitivePtr kPrimConv2DBackpropInput = std::make_shared<Primitive>("Conv2DBackpropInput");
inline const PrimitivePtr kPrimConv2DBackpropFilter = std::make_shared<Primitive>("Conv2DBackpropFilter");
inline const PrimitivePtr kPrimDepthwiseConv2dNative = std::make_shared<Primitive>("DepthwiseConv2dNative");
inline const PrimitivePtr kPrimDepthwiseConv2dNativeBackpropFilter =
  std::make_shared<Primitive>("DepthwiseConv2dNativeBackpropFilter");
inline const PrimitivePtr kPrimDepthwiseConv2dNativeBackpropInput =
  std::make_shared<Primitive>("DepthwiseConv2dNativeBackpropInput");
inline const PrimitivePtr kPrimBiasAddGrad = std::make_shared<Primitive>("BiasAddGrad");
inline const PrimitivePtr kPrimSoftmaxCrossEntropyWithLogits =
  std::make_shared<Primitive>("SoftmaxCrossEntropyWithLogits");
inline const PrimitivePtr kPrimSparseSoftmaxCrossEntropyWithLogits =
  std::make_shared<Primitive>("SparseSoftmaxCrossEntropyWithLogits");
inline const PrimitivePtr kPrimMomentum = std::make_shared<Primitive>("Momentum");
inline const PrimitivePtr kPrimApplyMomentum = std::make_shared<Primitive>("ApplyMomentum");
inline const PrimitivePtr kPrimLayerNorm = std::make_shared<Primitive>("LayerNorm");
inline const PrimitivePtr kPrimLayerNormGrad = std::make_shared<Primitive>("LayerNormGrad");
inline const PrimitivePtr kPrimLayerNormXBackprop = std::make_shared<Primitive>("LayerNormXBackprop");
inline const PrimitivePtr kPrimLayerNormBetaGammaBackprop = std::make_shared<Primitive>("LayerNormBetaGammaBackprop");
inline const PrimitivePtr kPrimDropoutGenMask = std::make_shared<Primitive>("DropoutGenMask");
inline const PrimitivePtr kPrimDropoutDoMask = std::make_shared<Primitive>("DropoutDoMask");
inline const PrimitivePtr kPrimOneHot = std::make_shared<Primitive>("OneHot");
inline const PrimitivePtr kPrimGelu = std::make_shared<Primitive>("Gelu");
inline const PrimitivePtr kPrimGeluGrad = std::make_shared<Primitive>("GeluGrad");
inline const PrimitivePtr kPrimRelu = std::make_shared<Primitive>("ReLU");
inline const PrimitivePtr kPrimReluV2 = std::make_shared<Primitive>("ReLUV2");
inline const PrimitivePtr kPrimZerosLike = std::make_shared<Primitive>("ZerosLike");
inline const PrimitivePtr kPrimBpropCut = std::make_shared<Primitive>("bprop_cut");
inline const PrimitivePtr kPrimFakeQuantPerLayer = std::make_shared<Primitive>("FakeQuantPerLayer");
inline const PrimitivePtr kPrimFakeQuantPerChannel = std::make_shared<Primitive>("FakeQuantPerChannel");
inline const PrimitivePtr kPrimApplyRMSProp = std::make_shared<Primitive>("ApplyRMSProp");

// Comm ops
inline const PrimitivePtr kPrimMirror = std::make_shared<Primitive>("_MirrorOperator");
inline const PrimitivePtr kPrimVirtualDiv = std::make_shared<Primitive>("_VirtualDiv");
inline const PrimitivePtr kPrimVirtualDataset = std::make_shared<Primitive>("_VirtualDataset");
inline const PrimitivePtr kPrimAllReduce = std::make_shared<Primitive>("AllReduce");

// RowTensor
inline const PrimitivePtr kPrimMakeRowTensor = std::make_shared<Primitive>("MakeRowTensor");
inline const PrimitivePtr kPrimRowTensorGetValues = std::make_shared<Primitive>("RowTensorGetValues");
inline const PrimitivePtr kPrimRowTensorGetIndices = std::make_shared<Primitive>("RowTensorGetIndices");
inline const PrimitivePtr kPrimRowTensorGetDenseShape = std::make_shared<Primitive>("RowTensorGetDenseShape");

// SparseTensor
inline const PrimitivePtr kPrimMakeSparseTensor = std::make_shared<Primitive>("MakeSparseTensor");
inline const PrimitivePtr kPrimSparseTensorGetValues = std::make_shared<Primitive>("SparseTensorGetValues");
inline const PrimitivePtr kPrimSparseTensorGetIndices = std::make_shared<Primitive>("SparseTensorGetIndices");
inline const PrimitivePtr kPrimSparseTensorGetDenseShape = std::make_shared<Primitive>("SparseTensorGetDenseShape");

// Maths
inline const PrimitivePtr kPrimTensorAdd = std::make_shared<Primitive>("TensorAdd");
inline const PrimitivePtr kPrimMatMul = std::make_shared<Primitive>("MatMul");
inline const PrimitivePtr kPrimBatchMatMul = std::make_shared<Primitive>("BatchMatMul");
inline const PrimitivePtr kPrimMaximumGrad = std::make_shared<Primitive>("MaximumGrad");
inline const PrimitivePtr kPrimMinimumGrad = std::make_shared<Primitive>("MinimumGrad");
inline const PrimitivePtr kPrimReduceMean = std::make_shared<Primitive>("ReduceMean");
inline const PrimitivePtr kPrimReduceSum = std::make_shared<Primitive>("ReduceSum");
inline const PrimitivePtr kPrimReduceAll = std::make_shared<Primitive>("ReduceAll");
inline const PrimitivePtr kPrimReduceAny = std::make_shared<Primitive>("ReduceAny");
inline const PrimitivePtr kPrimReduceMax = std::make_shared<Primitive>("ReduceMax");
inline const PrimitivePtr kPrimReduceMin = std::make_shared<Primitive>("ReduceMin");
inline const PrimitivePtr kPrimNeg = std::make_shared<Primitive>("Neg");
inline const PrimitivePtr kPrimSub = std::make_shared<Primitive>("Sub");
inline const PrimitivePtr kPrimMul = std::make_shared<Primitive>("Mul");
inline const PrimitivePtr kPrimMinimum = std::make_shared<Primitive>("Minimum");
inline const PrimitivePtr kPrimMaximum = std::make_shared<Primitive>("Maximum");
inline const PrimitivePtr kPrimSquare = std::make_shared<Primitive>("Square");
inline const PrimitivePtr kPrimCumSum = std::make_shared<Primitive>("CumSum");
inline const PrimitivePtr kPrimCumProd = std::make_shared<Primitive>("CumProd");
inline const PrimitivePtr kPrimSubscalar = std::make_shared<Primitive>("Subscalar");
inline const PrimitivePtr kPrimInplaceAdd = std::make_shared<Primitive>("InplaceAdd");
inline const PrimitivePtr kPrimInplaceSub = std::make_shared<Primitive>("InplaceSub");
inline const PrimitivePtr kPrimPow = std::make_shared<Primitive>("Pow");
inline const PrimitivePtr kPrimRealDiv = std::make_shared<Primitive>("RealDiv");
inline const PrimitivePtr kPrimSqrt = std::make_shared<Primitive>("Sqrt");
inline const PrimitivePtr kPrimReciprocal = std::make_shared<Primitive>("Reciprocal");
inline const PrimitivePtr kPrimExpandDims = std::make_shared<Primitive>("ExpandDims");

// Statements
inline const PrimitivePtr kPrimReturn = std::make_shared<Primitive>("return");
inline const PrimitivePtr kPrimSwitch = std::make_shared<Primitive>("switch");
inline const PrimitivePtr kPrimSwitchLayer = std::make_shared<Primitive>("switch_layer");
inline const PrimitivePtr kPrimAssign = std::make_shared<Primitive>("Assign");
inline const PrimitivePtr kPrimAssignAdd = std::make_shared<Primitive>("AssignAdd");
inline const PrimitivePtr kPrimAssignSub = std::make_shared<Primitive>("AssignSub");
inline const PrimitivePtr kPrimSelect = std::make_shared<Primitive>("Select");
inline const PrimitivePtr kPrimCall = std::make_shared<Primitive>("call");

inline const PrimitivePtr kPrimMakeTuple = std::make_shared<Primitive>("make_tuple");
inline const PrimitivePtr kPrimMakeSlice = std::make_shared<Primitive>("make_slice");
inline const PrimitivePtr kPrimTupleGetItem = std::make_shared<Primitive>("tuple_getitem");
inline const PrimitivePtr kPrimArrayGetItem = std::make_shared<Primitive>("array_getitem");
inline const PrimitivePtr kPrimTupleSetItem = std::make_shared<Primitive>("tuple_setitem");
inline const PrimitivePtr kPrimArraySetItem = std::make_shared<Primitive>("array_setitem");
inline const PrimitivePtr kPrimGetAttr = std::make_shared<Primitive>("getattr");
inline const PrimitivePtr kPrimTupleLen = std::make_shared<Primitive>("tuple_len");
inline const PrimitivePtr kPrimArrayLen = std::make_shared<Primitive>("array_len");
inline const PrimitivePtr kPrimTileShape = std::make_shared<Primitive>("tile_shape");
inline const PrimitivePtr kPrimGenerateShapeIndex = std::make_shared<Primitive>("generate_shape_index");
inline const PrimitivePtr kPrimGenerateInverseIndex = std::make_shared<Primitive>("generate_inverse_index");

// Debug ops
inline const PrimitivePtr kPrimScalarSummary = std::make_shared<Primitive>("ScalarSummary");
inline const PrimitivePtr kPrimImageSummary = std::make_shared<Primitive>("ImageSummary");
inline const PrimitivePtr kPrimTensorSummary = std::make_shared<Primitive>("TensorSummary");
inline const PrimitivePtr kPrimHistogramSummary = std::make_shared<Primitive>("HistogramSummary");
inline const PrimitivePtr kPrimDebug = std::make_shared<Primitive>("Debug");

// Other miscellaneous
inline const PrimitivePtr kPrimDepend = std::make_shared<Primitive>("Depend");
inline const PrimitivePtr kPrimPartial = std::make_shared<Primitive>("Partial");
inline const PrimitivePtr kPrimIdentity = std::make_shared<Primitive>("identity");
inline const PrimitivePtr kPrimHookBackward = std::make_shared<Primitive>("HookBackward");
inline const PrimitivePtr kPrimPrintShapeType = std::make_shared<Primitive>("PrintShapeType");
inline const PrimitivePtr kPrimSameTypeShape = std::make_shared<Primitive>("SameTypeShape");
inline const PrimitivePtr kPrimPrint = std::make_shared<Primitive>("Print");
inline const PrimitivePtr kPrimControlDepend = std::make_shared<Primitive>("ControlDepend");
inline const PrimitivePtr kPrimIs_ = std::make_shared<Primitive>("is_");
inline const PrimitivePtr kPrimIsNot = std::make_shared<Primitive>("is_not");
inline const PrimitivePtr kPrimInDict = std::make_shared<Primitive>("in_dict");
inline const PrimitivePtr kPrimNotInDict = std::make_shared<Primitive>("not_in_dict");
inline const PrimitivePtr kPrimIsConsant = std::make_shared<Primitive>("is_constant");
inline const PrimitivePtr kPrimEquivFormat = std::make_shared<Primitive>("EquivFormat");

// Structures
inline const PrimitivePtr kPrimMakeList = std::make_shared<Primitive>("make_list");
inline const PrimitivePtr kPrimMakeKeywordArg = std::make_shared<Primitive>("make_keyword_arg");
inline const PrimitivePtr kPrimListGetItem = std::make_shared<Primitive>("list_getitem");
inline const PrimitivePtr kPrimListSetItem = std::make_shared<Primitive>("list_setitem");
inline const PrimitivePtr kPrimDictGetItem = std::make_shared<Primitive>("dict_getitem");
inline const PrimitivePtr kPrimDictSetItem = std::make_shared<Primitive>("dict_setitem");
inline const PrimitivePtr kPrimListAppend = std::make_shared<Primitive>("list_append");
inline const PrimitivePtr kPrimListLen = std::make_shared<Primitive>("list_len");

// Other miscellaneous
inline const PrimitivePtr kPrimEnvSetItem = std::make_shared<Primitive>("env_setitem");
inline const PrimitivePtr kPrimEnvGetItem = std::make_shared<Primitive>("env_getitem");
inline const PrimitivePtr kPrimEnvAdd = std::make_shared<Primitive>("env_add");
inline const PrimitivePtr kPrimMakeRefKey = std::make_shared<Primitive>("MakeRefKey");
inline const PrimitivePtr kPrimGetRefKey = std::make_shared<Primitive>("get_ref_key");
inline const PrimitivePtr kPrimMakeRef = std::make_shared<Primitive>("make_ref");
inline const PrimitivePtr kPrimGetRefValue = std::make_shared<Primitive>("get_ref_value");

// Other primitve not used by backend but used in core;
inline const PrimitivePtr kPrimStateSetItem = std::make_shared<Primitive>("state_setitem");
inline const PrimitivePtr kPrimJ = std::make_shared<Primitive>("J");

// Used to build graph which have keyword arguments
inline const PrimitivePtr kPrimExtractKeywordArg = std::make_shared<Primitive>("extract_keyword_arg");
inline const PrimitivePtr kPrimMakeDict = std::make_shared<Primitive>("make_dict");

class DoSignaturePrimitive : public Primitive {
 public:
  explicit DoSignaturePrimitive(const std::string &name, const ValuePtr &function)
      : Primitive("S-Prim-" + name), function_(function) {}

  ~DoSignaturePrimitive() override = default;

  MS_DECLARE_PARENT(DoSignaturePrimitive, Primitive)

  const ValuePtr function() const { return function_; }

 private:
  ValuePtr function_;
};
using DoSignaturePrimitivePtr = std::shared_ptr<DoSignaturePrimitive>;
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPERATOR_OPS_H_
