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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPERATOR_OPS_H_
#define MINDSPORE_CCSRC_FRONTEND_OPERATOR_OPS_H_

#include <iostream>
#include <string>
#include <memory>
#include "ir/anf.h"
#include "ir/primitive.h"
#include "base/core_ops.h"

namespace mindspore {
// namespace to support primitive operators
namespace prim {
ValuePtr GetPythonOps(const std::string &op_name,
                      const std::string &module_name = "mindspore._extends.parse.standard_method",
                      bool use_signature = false);

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

// Type introspection
inline const PrimitivePtr kPrimTypeOf = std::make_shared<Primitive>("typeof");
inline const PrimitivePtr kPrimHasType = std::make_shared<Primitive>("hastype");

inline const PrimitivePtr kPrimDistribute = std::make_shared<Primitive>("distribute");
inline const PrimitivePtr kPrimDot = std::make_shared<Primitive>("dot");
inline const PrimitivePtr kPrimIm2Col = std::make_shared<Primitive>("im2col");
inline const PrimitivePtr kPrimCol2Im = std::make_shared<Primitive>("col2im");
inline const PrimitivePtr kPrimIm2ColV1 = std::make_shared<Primitive>("im2col_v1");
inline const PrimitivePtr kPrimCol2ImV1 = std::make_shared<Primitive>("col2im_v1");

inline const PrimitivePtr kPrimResolve = std::make_shared<Primitive>("resolve");
inline const PrimitivePtr kPrimEmbed = std::make_shared<Primitive>("embed");
inline const PrimitivePtr kPrimRefToEmbed = std::make_shared<Primitive>("RefToEmbed");
inline const PrimitivePtr kPrimCreateInstance = std::make_shared<Primitive>("create_instance");

inline const PrimitivePtr kPrimLabelGoto = std::make_shared<Primitive>("LabelGoto");
inline const PrimitivePtr kPrimLabelSwitch = std::make_shared<Primitive>("LabelSwitch");
inline const PrimitivePtr kPrimLabelSet = std::make_shared<Primitive>("LabelSet");

// Arrays
inline const PrimitivePtr kPrimScalarToArray = std::make_shared<Primitive>("scalar_to_array");
inline const PrimitivePtr kPrimArrayToScalar = std::make_shared<Primitive>("array_to_scalar");
inline const PrimitivePtr kPrimBroadcastShape = std::make_shared<Primitive>("broadcast_shape");
inline const PrimitivePtr kPrimArrayMap = std::make_shared<Primitive>("array_map");
inline const PrimitivePtr kPrimArrayReduce = std::make_shared<Primitive>("array_reduce");
inline const PrimitivePtr kPrimShape = std::make_shared<Primitive>("Shape");
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
inline const PrimitivePtr kPrimFakeBprop = std::make_shared<Primitive>("fake_bprop");
inline const PrimitivePtr kPrimBpropCut = std::make_shared<Primitive>("bprop_cut");
inline const PrimitivePtr kPrimFakeQuantPerLayer = std::make_shared<Primitive>("FakeQuantPerLayer");
inline const PrimitivePtr kPrimFakeQuantPerChannel = std::make_shared<Primitive>("FakeQuantPerChannel");
inline const PrimitivePtr kPrimApplyRMSProp = std::make_shared<Primitive>("ApplyRMSProp");

// Comm ops
inline const PrimitivePtr kPrimMirror = std::make_shared<Primitive>("_MirrorOperator");
inline const PrimitivePtr kPrimVirtualDiv = std::make_shared<Primitive>("_VirtualDiv");
inline const PrimitivePtr kPrimVirtualDataset = std::make_shared<Primitive>("_VirtualDataset");
inline const PrimitivePtr kPrimAllReduce = std::make_shared<Primitive>("AllReduce");

// IndexedSlices
inline const PrimitivePtr kPrimMakeIndexedSlices = std::make_shared<Primitive>("MakeIndexedSlices");
inline const PrimitivePtr kPrimIndexedSlicesGetValues = std::make_shared<Primitive>("IndexedSlicesGetValues");
inline const PrimitivePtr kPrimIndexedSlicesGetIndices = std::make_shared<Primitive>("IndexedSlicesGetIndices");
inline const PrimitivePtr kPrimIndexedSlicesGetDenseShape = std::make_shared<Primitive>("IndexedSlicesGetDenseShape");
inline const PrimitivePtr kPrimIsIndexedSlices = std::make_shared<Primitive>("IsIndexedSlices");

// SparseTensor
inline const PrimitivePtr kPrimMakeSparseTensor = std::make_shared<Primitive>("MakeSparseTensor");
inline const PrimitivePtr kPrimSparseTensorGetValues = std::make_shared<Primitive>("SparseTensorGetValues");
inline const PrimitivePtr kPrimSparseTensorGetIndices = std::make_shared<Primitive>("SparseTensorGetIndices");
inline const PrimitivePtr kPrimSparseTensorGetDenseShape = std::make_shared<Primitive>("SparseTensorGetDenseShape");

// attribute 'unroll_flag' of primitive 'switch', when 'unroll_flag' is '0', 'switch' will not unroll
const char SWITCH_UNROLL_FLAG[] = "unroll_flag";
// max loop count of for statement, when loop count is less then this value, the for loop will be unrolled, otherwise it
//  will be sunk(i.e. not unrolled)
const int MAX_FOR_LOOP_COUNT = 600;

class UnpackGraphPrimitive : public Primitive {
 public:
  explicit UnpackGraphPrimitive(const std::string &name, const bool &with_sens, const bool &need_unpack_args)
      : Primitive("UnpackGraph"), with_sens_in_args_(with_sens), need_unpack_args_(need_unpack_args) {}
  ~UnpackGraphPrimitive() override = default;
  MS_DECLARE_PARENT(UnpackGraphPrimitive, Primitive)
  bool with_sens_in_args() const { return with_sens_in_args_; }
  bool need_unpack_args() const { return need_unpack_args_; }

 private:
  bool with_sens_in_args_;
  bool need_unpack_args_;
};
using UnpackGraphPrimitivePtr = std::shared_ptr<UnpackGraphPrimitive>;
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_OPERATOR_OPS_H_
