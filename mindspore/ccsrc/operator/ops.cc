/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "operator/ops.h"
#include <memory>
#include <string>
#include "pipeline/parse/python_adapter.h"
#include "pipeline/parse/data_converter.h"

namespace mindspore {
// namespace to support primitive operators
namespace prim {
// Arithmetic
const PrimitivePtr kPrimScalarAdd = std::make_shared<Primitive>("scalar_add");
const PrimitivePtr kPrimScalarSub = std::make_shared<Primitive>("scalar_sub");
const PrimitivePtr kPrimScalarMul = std::make_shared<Primitive>("scalar_mul");
const PrimitivePtr kPrimScalarDiv = std::make_shared<Primitive>("scalar_div");
const PrimitivePtr kPrimScalarFloordiv = std::make_shared<Primitive>("scalar_floordiv");
const PrimitivePtr kPrimScalarMod = std::make_shared<Primitive>("scalar_mod");
const PrimitivePtr kPrimScalarPow = std::make_shared<Primitive>("scalar_pow");
const PrimitivePtr kPrimScalarTrunc = std::make_shared<Primitive>("scalar_trunc");
const PrimitivePtr kPrimScalarFloor = std::make_shared<Primitive>("scalar_floor");
const PrimitivePtr kPrimScalarUadd = std::make_shared<Primitive>("scalar_uadd");
const PrimitivePtr kPrimScalarUsub = std::make_shared<Primitive>("scalar_usub");
const PrimitivePtr kPrimScalarExp = std::make_shared<Primitive>("scalar_exp");
const PrimitivePtr kPrimScalarLog = std::make_shared<Primitive>("scalar_log");
const PrimitivePtr kPrimScalarSin = std::make_shared<Primitive>("scalar_sin");
const PrimitivePtr kPrimScalarCos = std::make_shared<Primitive>("scalar_cos");
const PrimitivePtr kPrimScalarTan = std::make_shared<Primitive>("scalar_tan");

// Comparisons
const PrimitivePtr kPrimScalarEq = std::make_shared<Primitive>("scalar_eq");
const PrimitivePtr kPrimScalarLt = std::make_shared<Primitive>("scalar_lt");
const PrimitivePtr kPrimScalarGt = std::make_shared<Primitive>("scalar_gt");
const PrimitivePtr kPrimScalarNe = std::make_shared<Primitive>("scalar_ne");
const PrimitivePtr kPrimScalarLe = std::make_shared<Primitive>("scalar_le");
const PrimitivePtr kPrimScalarGe = std::make_shared<Primitive>("scalar_ge");
const PrimitivePtr kPrimBoolNot = std::make_shared<Primitive>("bool_not");
const PrimitivePtr kPrimBoolAnd = std::make_shared<Primitive>("bool_and");
const PrimitivePtr kPrimBoolOr = std::make_shared<Primitive>("bool_or");
const PrimitivePtr kPrimBoolEq = std::make_shared<Primitive>("bool_eq");

// Type introspection
const PrimitivePtr kPrimTypeOf = std::make_shared<Primitive>("typeof");
const PrimitivePtr kPrimHasType = std::make_shared<Primitive>("hastype");

// Statements
const PrimitivePtr kPrimSwitch = std::make_shared<Primitive>("switch");
const PrimitivePtr kPrimSwitchLayer = std::make_shared<Primitive>("switch_layer");
const PrimitivePtr kPrimReturn = std::make_shared<Primitive>("return");
const PrimitivePtr kPrimAssign = std::make_shared<Primitive>("Assign");
const PrimitivePtr kPrimAssignAdd = std::make_shared<Primitive>("AssignAdd");
const PrimitivePtr kPrimAssignSub = std::make_shared<Primitive>("AssignSub");
const PrimitivePtr kPrimSelect = std::make_shared<Primitive>("Select");
const PrimitivePtr kPrimCall = std::make_shared<Primitive>("call");

const PrimitivePtr kPrimDistribute = std::make_shared<Primitive>("distribute");
const PrimitivePtr kPrimDot = std::make_shared<Primitive>("dot");
const PrimitivePtr kPrimIm2Col = std::make_shared<Primitive>("im2col");
const PrimitivePtr kPrimCol2Im = std::make_shared<Primitive>("col2im");
const PrimitivePtr kPrimIm2ColV1 = std::make_shared<Primitive>("im2col_v1");
const PrimitivePtr kPrimCol2ImV1 = std::make_shared<Primitive>("col2im_v1");

const PrimitivePtr kPrimResolve = std::make_shared<Primitive>("resolve");
const PrimitivePtr kPrimEmbed = std::make_shared<Primitive>("embed");
const PrimitivePtr kPrimRefToEmbed = std::make_shared<Primitive>("RefToEmbed");
const PrimitivePtr kPrimCreateInstance = std::make_shared<Primitive>("create_instance");

const PrimitivePtr kPrimLabelGoto = std::make_shared<Primitive>("LabelGoto");
const PrimitivePtr kPrimLabelSwitch = std::make_shared<Primitive>("LabelSwitch");
const PrimitivePtr kPrimLabelSet = std::make_shared<Primitive>("LabelSet");

// Structure
const PrimitivePtr kPrimStringEqual = std::make_shared<Primitive>("string_equal");
const PrimitivePtr kPrimStringConcat = std::make_shared<Primitive>("string_concat");
const PrimitivePtr kPrimMakeTuple = std::make_shared<Primitive>("make_tuple");
const PrimitivePtr kPrimMakeList = std::make_shared<Primitive>("make_list");
const PrimitivePtr kPrimMakeDict = std::make_shared<Primitive>("make_dict");
const PrimitivePtr kPrimMakeKeywordArg = std::make_shared<Primitive>("make_keyword_arg");
const PrimitivePtr kPrimExtractKeywordArg = std::make_shared<Primitive>("extract_keyword_arg");
const PrimitivePtr kPrimMakeSlice = std::make_shared<Primitive>("make_slice");
const PrimitivePtr kPrimMakeRecord = std::make_shared<Primitive>("make_record");
const PrimitivePtr kPrimTupleGetItem = std::make_shared<Primitive>("tuple_getitem");
const PrimitivePtr kPrimListGetItem = std::make_shared<Primitive>("list_getitem");
const PrimitivePtr kPrimArrayGetItem = std::make_shared<Primitive>("array_getitem");
const PrimitivePtr kPrimTupleSetItem = std::make_shared<Primitive>("tuple_setitem");
const PrimitivePtr kPrimListSetItem = std::make_shared<Primitive>("list_setitem");
const PrimitivePtr kPrimArraySetItem = std::make_shared<Primitive>("array_setitem");
const PrimitivePtr kPrimDictGetItem = std::make_shared<Primitive>("dict_getitem");
const PrimitivePtr kPrimDictSetItem = std::make_shared<Primitive>("dict_setitem");
const PrimitivePtr kPrimListAppend = std::make_shared<Primitive>("list_append");
const PrimitivePtr kPrimGetAttr = std::make_shared<Primitive>("getattr");
const PrimitivePtr kPrimTupleLen = std::make_shared<Primitive>("tuple_len");
const PrimitivePtr kPrimDictLen = std::make_shared<Primitive>("dict_len");
const PrimitivePtr kPrimListLen = std::make_shared<Primitive>("list_len");
const PrimitivePtr kPrimArrayLen = std::make_shared<Primitive>("array_len");
const PrimitivePtr kPrimListMap = std::make_shared<Primitive>("list_map");
const PrimitivePtr kPrimListReduce = std::make_shared<Primitive>("list_reduce");
const PrimitivePtr kPrimTupleReversed = std::make_shared<Primitive>("tuple_reversed");

const PrimitivePtr kPrimTileShape = std::make_shared<Primitive>("tile_shape");
const PrimitivePtr kPrimReducedShape = std::make_shared<Primitive>("reduced_shape");
const PrimitivePtr kPrimTupleDiv = std::make_shared<Primitive>("tuple_div");
const PrimitivePtr kPrimTupleToArray = std::make_shared<Primitive>("tuple_to_array");
const PrimitivePtr kPrimShapeMul = std::make_shared<Primitive>("shape_mul");
const PrimitivePtr kPrimGenerateShapeIndex = std::make_shared<Primitive>("generate_shape_index");
const PrimitivePtr kPrimGenerateInverseIndex = std::make_shared<Primitive>("generate_inverse_index");
const PrimitivePtr kPrimTupleEqual = std::make_shared<Primitive>("tuple_equal");
const PrimitivePtr kPrimListEqual = std::make_shared<Primitive>("list_equal");
const PrimitivePtr kPrimMakeRange = std::make_shared<Primitive>("make_range");
const PrimitivePtr kPrimStopGradient = std::make_shared<Primitive>("stop_gradient");

// Arrays
const PrimitivePtr kPrimScalarToArray = std::make_shared<Primitive>("scalar_to_array");
const PrimitivePtr kPrimArrayToScalar = std::make_shared<Primitive>("array_to_scalar");
const PrimitivePtr kPrimBroadcastShape = std::make_shared<Primitive>("broadcast_shape");
const PrimitivePtr kPrimArrayMap = std::make_shared<Primitive>("array_map");
const PrimitivePtr kPrimArrayReduce = std::make_shared<Primitive>("array_reduce");
const PrimitivePtr kPrimShape = std::make_shared<Primitive>("Shape");
const PrimitivePtr kPrimCast = std::make_shared<Primitive>("Cast");
const PrimitivePtr kPrimConcat = std::make_shared<Primitive>("Concat");
const PrimitivePtr kPrimSqueeze = std::make_shared<Primitive>("Squeeze");
const PrimitivePtr kPrimTranspose = std::make_shared<Primitive>("Transpose");
const PrimitivePtr kPrimGatherV2 = std::make_shared<Primitive>("GatherV2");
const PrimitivePtr kPrimSize = std::make_shared<Primitive>("Size");
const PrimitivePtr kPrimArgMax = std::make_shared<Primitive>("Argmax");
const PrimitivePtr kPrimPack = std::make_shared<Primitive>("Pack");
const PrimitivePtr kPrimUnsortedSegmentSum = std::make_shared<Primitive>("UnsortedSegmentSum");
const PrimitivePtr kPrimUnsortedSegmentMin = std::make_shared<Primitive>("UnsortedSegmentMin");
const PrimitivePtr kPrimConcatOffset = std::make_shared<Primitive>("ConcatOffset");
const PrimitivePtr kPrimReshape = std::make_shared<Primitive>("Reshape");
const PrimitivePtr kPrimTile = std::make_shared<Primitive>("Tile");
const PrimitivePtr kPrimAddN = std::make_shared<Primitive>("AddN");
const PrimitivePtr KPrimTransData = std::make_shared<Primitive>("TransData");
const PrimitivePtr kPrimNMSWithMask = std::make_shared<Primitive>("NMSWithMask");
const PrimitivePtr kPrimPad = std::make_shared<Primitive>("Pad");

// Maths
const PrimitivePtr kPrimTensorAdd = std::make_shared<Primitive>("TensorAdd");
const PrimitivePtr kPrimMatMul = std::make_shared<Primitive>("MatMul");
const PrimitivePtr kPrimBatchMatMul = std::make_shared<Primitive>("BatchMatMul");
const PrimitivePtr kPrimMaximumGrad = std::make_shared<Primitive>("MaximumGrad");
const PrimitivePtr kPrimMinimumGrad = std::make_shared<Primitive>("MinimumGrad");
const PrimitivePtr kPrimReduceMean = std::make_shared<Primitive>("ReduceMean");
const PrimitivePtr kPrimReduceSum = std::make_shared<Primitive>("ReduceSum");
const PrimitivePtr kPrimReduceAll = std::make_shared<Primitive>("ReduceAll");
const PrimitivePtr kPrimReduceMax = std::make_shared<Primitive>("ReduceMax");
const PrimitivePtr kPrimReduceMin = std::make_shared<Primitive>("ReduceMin");
const PrimitivePtr kPrimNeg = std::make_shared<Primitive>("Neg");
const PrimitivePtr kPrimSub = std::make_shared<Primitive>("Sub");
const PrimitivePtr kPrimMul = std::make_shared<Primitive>("Mul");
const PrimitivePtr kPrimMinimum = std::make_shared<Primitive>("Minimum");
const PrimitivePtr kPrimMaximum = std::make_shared<Primitive>("Maximum");
const PrimitivePtr kPrimSquare = std::make_shared<Primitive>("Square");
const PrimitivePtr kPrimEqual = std::make_shared<Primitive>("Equal");
const PrimitivePtr kPrimLess = std::make_shared<Primitive>("Less");
const PrimitivePtr kPrimLessEqual = std::make_shared<Primitive>("LessEqual");
const PrimitivePtr kPrimCumSum = std::make_shared<Primitive>("CumSum");
const PrimitivePtr kPrimCumProd = std::make_shared<Primitive>("CumProd");

// NN
const PrimitivePtr kPrimFlatten = std::make_shared<Primitive>("Flatten");
const PrimitivePtr kPrimLogSoftmax = std::make_shared<Primitive>("LogSoftmax");
const PrimitivePtr kPrimLogSoftmaxGrad = std::make_shared<Primitive>("LogSoftmaxGrad");
const PrimitivePtr kPrimTanh = std::make_shared<Primitive>("Tanh");
const PrimitivePtr kPrimTanhGrad = std::make_shared<Primitive>("TanhGrad");
const PrimitivePtr kPrimPooling = std::make_shared<Primitive>("Pooling");
const PrimitivePtr kPrimPoolingGrad = std::make_shared<Primitive>("PoolingGrad");
const PrimitivePtr kPrimMaxPool = std::make_shared<Primitive>("MaxPool");
const PrimitivePtr kPrimMaxPoolGrad = std::make_shared<Primitive>("MaxPoolGrad");
const PrimitivePtr kPrimAvgPoolGrad = std::make_shared<Primitive>("AvgPoolGrad");
const PrimitivePtr kPrimFusedBatchNorm = std::make_shared<Primitive>("FusedBatchNorm");
const PrimitivePtr kPrimConv2D = std::make_shared<Primitive>("Conv2D");
const PrimitivePtr kPrimFusedBatchNormGrad = std::make_shared<Primitive>("FusedBatchNormGrad");
const PrimitivePtr kPrimBatchNorm = std::make_shared<Primitive>("BatchNorm");
const PrimitivePtr kPrimBatchNormGrad = std::make_shared<Primitive>("BatchNormGrad");
const PrimitivePtr kPrimReluGrad = std::make_shared<Primitive>("ReluGrad");
const PrimitivePtr kPrimConv2DBackpropInput = std::make_shared<Primitive>("Conv2DBackpropInput");
const PrimitivePtr kPrimConv2DBackpropFilter = std::make_shared<Primitive>("Conv2DBackpropFilter");
const PrimitivePtr kPrimDepthwiseConv2dNative = std::make_shared<Primitive>("DepthwiseConv2dNative");
const PrimitivePtr kPrimDepthwiseConv2dNativeBackpropFilter =
  std::make_shared<Primitive>("DepthwiseConv2dNativeBackpropFilter");
const PrimitivePtr kPrimDepthwiseConv2dNativeBackpropInput =
  std::make_shared<Primitive>("DepthwiseConv2dNativeBackpropInput");
const PrimitivePtr kPrimBiasAddGrad = std::make_shared<Primitive>("BiasAddGrad");
const PrimitivePtr kPrimSoftmaxCrossEntropyWithLogits = std::make_shared<Primitive>("SoftmaxCrossEntropyWithLogits");
const PrimitivePtr kPrimSparseSoftmaxCrossEntropyWithLogits =
  std::make_shared<Primitive>("SparseSoftmaxCrossEntropyWithLogits");
const PrimitivePtr kPrimMomentum = std::make_shared<Primitive>("Momentum");
const PrimitivePtr kPrimApplyMomentum = std::make_shared<Primitive>("ApplyMomentum");
const PrimitivePtr kPrimLayerNorm = std::make_shared<Primitive>("LayerNorm");
const PrimitivePtr kPrimLayerNormGrad = std::make_shared<Primitive>("LayerNormGrad");
const PrimitivePtr kPrimLayerNormXBackprop = std::make_shared<Primitive>("LayerNormXBackprop");
const PrimitivePtr kPrimLayerNormBetaGammaBackprop = std::make_shared<Primitive>("LayerNormBetaGammaBackprop");
const PrimitivePtr kPrimDropoutGenMask = std::make_shared<Primitive>("DropoutGenMask");
const PrimitivePtr kPrimDropoutDoMask = std::make_shared<Primitive>("DropoutDoMask");
const PrimitivePtr kPrimOneHot = std::make_shared<Primitive>("OneHot");
const PrimitivePtr kPrimGelu = std::make_shared<Primitive>("Gelu");
const PrimitivePtr kPrimGeluGrad = std::make_shared<Primitive>("GeluGrad");
const PrimitivePtr kPrimRelu = std::make_shared<Primitive>("ReLU");
const PrimitivePtr kPrimReluV2 = std::make_shared<Primitive>("ReLUV2");
const PrimitivePtr kPrimZerosLikeTensor = std::make_shared<Primitive>("zeros_like_tensor");
const PrimitivePtr kPrimFakeBprop = std::make_shared<Primitive>("fake_bprop");
const PrimitivePtr kPrimBpropCut = std::make_shared<Primitive>("bprop_cut");

// Other miscellaneous
const PrimitivePtr kPrimIdentity = std::make_shared<Primitive>("identity");
const PrimitivePtr kPrimPartial = std::make_shared<Primitive>("partial");
const PrimitivePtr kPrimJ = std::make_shared<Primitive>("J");
const PrimitivePtr kPrimEnvSetItem = std::make_shared<Primitive>("env_setitem");
const PrimitivePtr kPrimEnvGetItem = std::make_shared<Primitive>("env_getitem");
const PrimitivePtr kPrimEnvAdd = std::make_shared<Primitive>("env_add");
const PrimitivePtr kPrimMakeRefKey = std::make_shared<Primitive>("MakeRefKey");
const PrimitivePtr kPrimGetRefKey = std::make_shared<Primitive>("get_ref_key");
const PrimitivePtr kPrimGetRefValue = std::make_shared<Primitive>("get_ref_value");
const PrimitivePtr kPrimGetRefOrigin = std::make_shared<Primitive>("get_ref_origin");
const PrimitivePtr kPrimInsertGradientOf = std::make_shared<Primitive>("InsertGradientOf");
const PrimitivePtr kPrimHookBackward = std::make_shared<Primitive>("HookBackward");
const PrimitivePtr kPrimPrintShapeType = std::make_shared<Primitive>("PrintShapeType");
const PrimitivePtr kPrimSameTypeShape = std::make_shared<Primitive>("SameTypeShape");
const PrimitivePtr kPrimCheckBprop = std::make_shared<Primitive>("CheckBprop");
const PrimitivePtr kPrimPrint = std::make_shared<Primitive>("Print");

const PrimitivePtr kPrimMakeRef = std::make_shared<Primitive>("make_ref");
const PrimitivePtr kPrimDepend = std::make_shared<Primitive>("depend");
const PrimitivePtr kPrimStateSetItem = std::make_shared<Primitive>("state_setitem");

const PrimitivePtr kPrimBroadcastGradientArgs = std::make_shared<Primitive>("BroadcastGradientArgs");
const PrimitivePtr kPrimControlDepend = std::make_shared<Primitive>("ControlDepend");
const PrimitivePtr kPrimIs_ = std::make_shared<Primitive>("is_");
const PrimitivePtr kPrimIsNot = std::make_shared<Primitive>("is_not");
const PrimitivePtr kPrimInDict = std::make_shared<Primitive>("in_dict");
const PrimitivePtr kPrimNotInDict = std::make_shared<Primitive>("not_in_dict");

// Comm ops
const PrimitivePtr kPrimMirror = std::make_shared<Primitive>("_MirrorOperator");
const PrimitivePtr kPrimVirtualDiv = std::make_shared<Primitive>("_VirtualDiv");
const PrimitivePtr kPrimVirtualDataset = std::make_shared<Primitive>("_VirtualDataset");
const PrimitivePtr kPrimAllReduce = std::make_shared<Primitive>("AllReduce");

// Debug ops
const PrimitivePtr kPrimScalarSummary = std::make_shared<Primitive>("ScalarSummary");
const PrimitivePtr kPrimImageSummary = std::make_shared<Primitive>("ImageSummary");
const PrimitivePtr kPrimTensorSummary = std::make_shared<Primitive>("TensorSummary");
const PrimitivePtr kPrimHistogramSummary = std::make_shared<Primitive>("HistogramSummary");

ValuePtr GetPythonOps(const std::string &op_name, const std::string &module_name) {
  py::object obj = parse::python_adapter::GetPyFn(module_name, op_name);
  ValuePtr node = nullptr;
  bool succ = parse::ConvertData(obj, &node);
  if (!succ) {
    MS_LOG(EXCEPTION) << "get Python op " << op_name << " from " << module_name << " fail";
  }
  return node;
}
}  // namespace prim
}  // namespace mindspore
