/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PREDICTOUTTYPEMAP_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PREDICTOUTTYPEMAP_H_

#include <string>
#include <memory>

#include "utils/hash_map.h"
#include "ops/base_operator.h"
#include "base/base.h"

namespace mindspore {
namespace pynative {
using PredictOutTypeMap = mindspore::HashMap<std::string, TypePtr>;
const TypePtr kTupleTensor2 = std::make_shared<Tuple>(TypePtrList{kTensorType, kTensorType});
const TypePtr kTupleTensor3 = std::make_shared<Tuple>(TypePtrList{kTensorType, kTensorType, kTensorType});
const TypePtr kTupleTensor4 = std::make_shared<Tuple>(TypePtrList{kTensorType, kTensorType, kTensorType, kTensorType});
const TypePtr kTupleTensor5 =
  std::make_shared<Tuple>(TypePtrList{kTensorType, kTensorType, kTensorType, kTensorType, kTensorType});
const TypePtr kTupleTensor6 =
  std::make_shared<Tuple>(TypePtrList{kTensorType, kTensorType, kTensorType, kTensorType, kTensorType, kTensorType});
const TypePtr kTupleTensor7 = std::make_shared<Tuple>(
  TypePtrList{kTensorType, kTensorType, kTensorType, kTensorType, kTensorType, kTensorType, kTensorType});
const TypePtr kTupleTensor8 = std::make_shared<Tuple>(
  TypePtrList{kTensorType, kTensorType, kTensorType, kTensorType, kTensorType, kTensorType, kTensorType, kTensorType});
const TypePtr kTupleTensor9 = std::make_shared<Tuple>(TypePtrList{
  kTensorType, kTensorType, kTensorType, kTensorType, kTensorType, kTensorType, kTensorType, kTensorType, kTensorType});

// If Abstract of the operator is constant, please add the type kTypeNone, such as "Size".
// If Abstract of the operator is Tensor and Tuple[Tensor], please add the type kAnyType, such as "Eigh".
inline static PredictOutTypeMap out_type_prediction = {{"ActsULQ", kTupleTensor4},
                                                       {"Adam", kTupleTensor3},
                                                       {"AdamApplyOne", kTupleTensor3},
                                                       {"AdamApplyOneAssign", kTupleTensor3},
                                                       {"AdamApplyOneWithDecay", kTupleTensor3},
                                                       {"AdamApplyOneWithDecayAssign", kTupleTensor3},
                                                       {"AdamWeightDecay", kTupleTensor3},
                                                       {"AdaptiveMaxPool2D", kTupleTensor2},
                                                       {"AdaptiveMaxPool3D", kTupleTensor2},
                                                       {"ApplyAdaMax", kTupleTensor3},
                                                       {"ApplyAdadelta", kTupleTensor3},
                                                       {"ApplyAdagrad", kTupleTensor2},
                                                       {"ApplyAdagradDA", kTupleTensor3},
                                                       {"ApplyAdagradV2", kTupleTensor2},
                                                       {"ApplyAdamWithAmsgrad", kTupleTensor4},
                                                       {"ApplyAddSign", kTupleTensor2},
                                                       {"ApplyKerasMomentum", kTupleTensor2},
                                                       {"ApplyPowerSign", kTupleTensor2},
                                                       {"ApplyProximalAdagrad", kTupleTensor2},
                                                       {"ArgMaxWithValue", kTupleTensor2},
                                                       {"ArgMinWithValue", kTupleTensor2},
                                                       {"BNTrainingReduce", kTupleTensor2},
                                                       {"BNTrainingUpdate", kTupleTensor5},
                                                       {"BNTrainingUpdateGrad", kTupleTensor2},
                                                       {"BNTrainingUpdateV2", kTupleTensor3},
                                                       {"BNTrainingUpdateV3", kTupleTensor5},
                                                       {"BasicLSTMCell", kTupleTensor7},
                                                       {"BasicLSTMCellCStateGrad", kTupleTensor2},
                                                       {"BasicLSTMCellCStateGradV2", kTupleTensor2},
                                                       {"BasicLSTMCellInputGrad", kTupleTensor2},
                                                       {"BasicLSTMCellWeightGrad", kTupleTensor2},
                                                       {"BatchNorm", kTupleTensor5},
                                                       {"BatchNormFold2GradD", kTupleTensor4},
                                                       {"BatchNormFold2GradReduce", kTupleTensor2},
                                                       {"BatchNormFoldD", kTupleTensor7},
                                                       {"BatchNormGrad", kTupleTensor3},
                                                       {"BatchNormGradWithActivation", kTupleTensor3},
                                                       {"BatchNormGradWithAddAndActivation", kTupleTensor4},
                                                       {"BatchNormGradGrad", kTupleTensor3},
                                                       {"BiasDropoutAdd", kTupleTensor2},
                                                       {"Broadcast", kTuple},
                                                       {"CSRSparseMatrixToSparseTensor", kTupleTensor3},
                                                       {"CTCGreedyDecoder", kTupleTensor4},
                                                       {"CTCLoss", kTupleTensor2},
                                                       {"CTCLossV2", kTupleTensor2},
                                                       {"Coalesce", kTupleTensor3},
                                                       {"ConcatOffset", kTypeNone},
                                                       {"CombinedNonMaxSuppression", kTupleTensor4},
                                                       {"ComputeAccidentalHits", kTupleTensor3},
                                                       {"ConfusionMulGrad", kTupleTensor2},
                                                       {"CorrectionMulGradReduce", kTupleTensor3},
                                                       {"CudnnGRU", kTupleTensor4},
                                                       {"Cummax", kTupleTensor2},
                                                       {"Cummin", kTupleTensor2},
                                                       {"Custom", kAnyType},
                                                       {"DSDGrad", kTupleTensor3},
                                                       {"DeformableOffsetsGrad", kTupleTensor2},
                                                       {"DenseToCSRSparseMatrix", kTupleTensor5},
                                                       {"DenseToDenseSetOperation", kTupleTensor3},
                                                       {"DenseToSparseSetOperation", kTupleTensor3},
                                                       {"Dropout", kTupleTensor2},
                                                       {"Dropout2D", kTupleTensor2},
                                                       {"Dropout3D", kTupleTensor2},
                                                       {"DynamicBroadcastGradientArgs", kTupleTensor2},
                                                       {"DynamicGRUV2", kTupleTensor6},
                                                       {"DynamicGRUV2Grad", kTupleTensor6},
                                                       {"DynamicRNN", kTupleTensor8},
                                                       {"Eig", kTupleTensor2},
                                                       {"Eigh", kAnyType},
                                                       {"FakeLearnedScaleQuantPerChannelGradD", kTupleTensor2},
                                                       {"FakeLearnedScaleQuantPerLayerGradD", kTupleTensor2},
                                                       {"FakeQuantWithMinMaxVarsGradient", kTupleTensor3},
                                                       {"FakeQuantWithMinMaxVarsPerChannelGradient", kTupleTensor3},
                                                       {"FractionalAvgPool", kTupleTensor3},
                                                       {"FractionalMaxPool", kTupleTensor3},
                                                       {"FractionalMaxPool3DWithFixedKsize", kTupleTensor2},
                                                       {"FractionalMaxPoolWithFixedKsize", kTupleTensor2},
                                                       {"FusedDbnDw", kTupleTensor2},
                                                       {"FusedMulAddNL2loss", kTupleTensor2},
                                                       {"FusedMulApplyMomentum", kTupleTensor2},
                                                       {"FusedMulApplyMomentumExtern", kTupleTensor3},
                                                       {"FusedSparseAdam", kTupleTensor3},
                                                       {"FusedSparseFtrl", kTupleTensor3},
                                                       {"FusedSparseLazyAdam", kTupleTensor3},
                                                       {"FusedSparseProximalAdagrad", kTupleTensor2},
                                                       {"GRU", kTupleTensor4},
                                                       {"GRUV2", kTupleTensor4},
                                                       {"GRUV2Grad", kTupleTensor3},
                                                       {"GRUV2HiddenGrad", kTupleTensor3},
                                                       {"GRUV2HiddenGradCell", kTupleTensor3},
                                                       {"Geqrf", kTupleTensor2},
                                                       {"GetNext", kTuple},
                                                       {"GridSampler2DGrad", kTupleTensor2},
                                                       {"GridSampler3DGrad", kTupleTensor2},
                                                       {"IFMR", kTupleTensor2},
                                                       {"IdentityN", kTuple},
                                                       {"InstanceNorm", kTupleTensor3},
                                                       {"InstanceNormGrad", kTupleTensor3},
                                                       {"InstanceNormV2", kTupleTensor3},
                                                       {"InstanceNormV2Grad", kTupleTensor3},
                                                       {"InvertPermutation", kTypeNone},
                                                       {"LSTM", kTupleTensor5},
                                                       {"LSTMGrad", kTupleTensor4},
                                                       {"LSTMGradData", kTupleTensor3},
                                                       {"LSTMInputGrad", kTupleTensor4},
                                                       {"LSTMV2", kTupleTensor5},
                                                       {"LU", kTupleTensor3},
                                                       {"LambApplyOptimizerAssign", kTupleTensor3},
                                                       {"LambNextMV", kTupleTensor4},
                                                       {"LambNextMVWithDecay", kTupleTensor4},
                                                       {"LambNextRight", kTupleTensor2},
                                                       {"LayerNorm", kTupleTensor3},
                                                       {"LayerNormBetaGammaBackprop", kTupleTensor2},
                                                       {"LayerNormBetaGammaBackpropV2", kTupleTensor2},
                                                       {"LayerNormGrad", kTupleTensor3},
                                                       {"LayerNormGradGrad", kTupleTensor3},
                                                       {"LayerNormXBackpropV2", kTupleTensor2},
                                                       {"LinearSumAssignment", kTupleTensor2},
                                                       {"ListDiff", kTupleTensor2},
                                                       {"LogMatrixDeterminant", kTupleTensor2},
                                                       {"LogUniformCandidateSampler", kTupleTensor3},
                                                       {"Lu", kTupleTensor2},
                                                       {"LuUnpack", kTupleTensor3},
                                                       {"LuUnpackGrad", kTupleTensor2},
                                                       {"MakeTuple", kTypeNone},
                                                       {"MapCacheIdx", kTupleTensor4},
                                                       {"MapTensorGetData", kTupleTensor2},
                                                       {"MatmulDDS", kTupleTensor2},
                                                       {"MatmulDDSGrad", kTupleTensor2},
                                                       {"MaxPool3DWithArgmax", kTupleTensor2},
                                                       {"MaxPoolWithArgmax", kTupleTensor2},
                                                       {"MaxPoolWithArgmaxV2", kTupleTensor2},
                                                       {"MaximumGrad", kTupleTensor2},
                                                       {"MaximumGradGrad", kTupleTensor3},
                                                       {"Median", kTupleTensor2},
                                                       {"MedianGrad", kTupleTensor2},
                                                       {"Meshgrid", kTuple},
                                                       {"MinMaxUpdatePerChannel", kTupleTensor2},
                                                       {"MinMaxUpdatePerLayer", kTupleTensor2},
                                                       {"MinimumGrad", kTupleTensor2},
                                                       {"MinimumGradGrad", kTupleTensor3},
                                                       {"MultilabelMarginLoss", kTupleTensor2},
                                                       {"NLLLoss", kTupleTensor2},
                                                       {"NMSWithMask", kTupleTensor3},
                                                       {"PReLUGrad", kTupleTensor2},
                                                       {"PSROIPooling", kAnyType},
                                                       {"PriorityReplayBufferSample", kTuple},
                                                       {"Qr", kTupleTensor2},
                                                       {"RNNTLoss", kTupleTensor2},
                                                       {"RaggedRange", kTupleTensor2},
                                                       {"RaggedTensorToSparse", kTupleTensor3},
                                                       {"RandomChoiceWithMask", kTupleTensor2},
                                                       {"ReLUV2", kTupleTensor2},
                                                       {"ReduceStd", kTupleTensor2},
                                                       {"ReservoirReplayBufferDestroy", kTupleTensor4},
                                                       {"SampleDistortedBoundingBoxV2", kTupleTensor3},
                                                       {"ScalarAdd", kTypeNone},
                                                       {"ScalarBool", kTypeNone},
                                                       {"ScalarDiv", kTypeNone},
                                                       {"ScalarFloordiv", kTypeNone},
                                                       {"ScalarMod", kTypeNone},
                                                       {"ScalarMul", kTypeNone},
                                                       {"ScalarSub", kTypeNone},
                                                       {"SelfAdjointEig", kTupleTensor2},
                                                       {"SequenceAdd", kTypeNone},
                                                       {"SequenceAddN", kTypeNone},
                                                       {"SequenceCount", kTypeNone},
                                                       {"SequenceIndex", kTypeNone},
                                                       {"SequenceMul", kTypeNone},
                                                       {"SequenceMax", kTypeNone},
                                                       {"SequenceMin", kTypeNone},
                                                       {"SequenceSlice", kTypeNone},
                                                       {"SequenceSliceGrad", kTypeNone},
                                                       {"SequenceSliceSetItem", kTypeNone},
                                                       {"SequenceZerosLike", kTypeNone},
                                                       {"Size", kTypeNone},
                                                       {"SoftmaxCrossEntropyWithLogits", kTupleTensor2},
                                                       {"SoftmaxV2WithDropoutDoMaskV3", kTupleTensor2},
                                                       {"Sort", kTupleTensor2},
                                                       {"SparseAdd", kTupleTensor3},
                                                       {"SparseAddGrad", kTupleTensor2},
                                                       {"SparseApplyAdadelta", kTupleTensor3},
                                                       {"SparseApplyAdagrad", kTupleTensor2},
                                                       {"SparseApplyAdagradV2", kTupleTensor2},
                                                       {"SparseApplyFtrl", kTupleTensor3},
                                                       {"SparseApplyFtrlV2", kTupleTensor3},
                                                       {"SparseApplyProximalAdagrad", kTupleTensor2},
                                                       {"SparseApplyRMSProp", kTupleTensor3},
                                                       {"SparseConcat", kTupleTensor3},
                                                       {"SparseCountSparseOutput", kTupleTensor3},
                                                       {"SparseCross", kTupleTensor3},
                                                       {"SparseFillEmptyRows", kTupleTensor4},
                                                       {"SparseFillEmptyRowsGrad", kTupleTensor2},
                                                       {"SparseMatrixAdd", kTupleTensor5},
                                                       {"SparseMatrixMul", kTupleTensor5},
                                                       {"SparseMatrixSoftmax", kTupleTensor5},
                                                       {"SparseMatrixSparseMatMul", kTupleTensor5},
                                                       {"SparseMatrixTranspose", kTupleTensor5},
                                                       {"SparseReorder", kTupleTensor2},
                                                       {"SparseReshape", kTupleTensor2},
                                                       {"SparseSlice", kTupleTensor3},
                                                       {"SparseSoftmaxCrossEntropyWithLogitsV2", kTupleTensor2},
                                                       {"SparseSparseMaximum", kTupleTensor2},
                                                       {"SparseSparseMinimum", kTupleTensor2},
                                                       {"SparseSplit", kTuple},
                                                       {"SparseTensorToCSRSparseMatrix", kTupleTensor5},
                                                       {"Split", kTuple},
                                                       {"SquareSumAll", kTupleTensor2},
                                                       {"SquareSumV2", kTupleTensor2},
                                                       {"Sspaddmm", kTupleTensor3},
                                                       {"SubAndFilter", kTupleTensor2},
                                                       {"Svd", kTupleTensor3},
                                                       {"TensorToList", kTypeNone},
                                                       {"TensorToScalar", kTypeNone},
                                                       {"TensorToTuple", kTypeNone},
                                                       {"TopK", kTupleTensor2},
                                                       {"TupleGetItem", kTypeNone},
                                                       {"UniformCandidateSampler", kTupleTensor3},
                                                       {"Unique", kTupleTensor2},
                                                       {"UniqueConsecutive", kTupleTensor3},
                                                       {"UniqueWithPad", kTupleTensor2},
                                                       {"Unpack", kTuple},
                                                       {"Unstack", kTuple},
                                                       {"bit_and", kTypeNone},
                                                       {"bit_or", kTypeNone},
                                                       {"make_range", kTypeNone},
                                                       {"scalar_eq", kTypeNone},
                                                       {"scalar_ge", kTypeNone},
                                                       {"scalar_gt", kTypeNone},
                                                       {"scalar_le", kTypeNone},
                                                       {"scalar_lt", kTypeNone},
                                                       {"sequence_len", kTypeNone},
                                                       {"tuple_setitem", kTypeNone}};

static TypePtr PredictOutTypeByName(const std::string &op_name) {
  static PredictOutTypeMap ops_map;
  const auto iter = ops_map.find(op_name);
  if (iter != ops_map.end()) {
    return iter->second;
  }
  static auto operator_fns = ops::OperatorRegister::GetInstance().GetOperatorMap();
  if (operator_fns.find(op_name) == operator_fns.end()) {
    return ops_map[op_name] = kTypeNone;
  }
  const auto pre_iter = out_type_prediction.find(op_name);
  auto type = pre_iter == out_type_prediction.end() ? kTensorType : pre_iter->second;
  return ops_map[op_name] = type;
}
}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PREDICTOUTTYPEMAP_H_
