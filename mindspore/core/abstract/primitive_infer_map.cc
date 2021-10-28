/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "abstract/primitive_infer_map.h"
#include <map>
#include <string>
#include <vector>
#include "ops/exp.h"
#include "ops/log.h"
#include "ops/reciprocal.h"
#include "ops/real_div.h"
#include "ops/add.h"
#include "ops/equal.h"
#include "ops/not_equal.h"
#include "ops/neg.h"
#include "ops/mul.h"
#include "ops/sub.h"
#include "ops/strided_slice.h"
#include "ops/reduce_sum.h"
#include "abstract/abstract_function.h"
#include "abstract/infer_functions.h"
#include "utils/ms_context.h"
#include "ops/tile.h"

namespace mindspore {
namespace abstract {
std::vector<int64_t> GetDependsFormMap(const CNodePtr &cnode) {
  const auto kOneHot = prim::kPrimOneHot->name();
  const auto kDropoutGenMask = prim::kPrimDropoutGenMask->name();
  const auto kTranspose = prim::kPrimTranspose->name();
  const auto kReduceSum = prim::kPrimReduceSum->name();
  const auto kUnsortedSegmentSum = prim::kPrimUnsortedSegmentSum->name();
  const auto kUnsortedSegmentMin = prim::kPrimUnsortedSegmentMin->name();
  const auto kUnsortedSegmentMax = prim::kPrimUnsortedSegmentMax->name();
  const auto kGather = prim::kPrimGather->name();
  const auto kGatherV2 = prim::kPrimGatherV2->name();
  const auto kDynamicShape = prim::kPrimDynamicShape->name();
  const auto kRange = prim::kPrimRange->name();
  const auto kConv2DBackpropFilter = prim::kPrimConv2DBackpropFilter->name();
  const auto kConv2DBackpropInput = prim::kPrimConv2DBackpropInput->name();
  // common dynamic shape depends
  static std::map<std::string, std::vector<int64_t>> dynamic_shape_depends = {{kUnsortedSegmentSum, {2}},
                                                                              {kUnsortedSegmentMin, {2}},
                                                                              {kUnsortedSegmentMax, {2}},
                                                                              {kGather, {2}},
                                                                              {kGatherV2, {2}},
                                                                              {kDynamicShape, {0}},
                                                                              {kRange, {0, 1, 2}},
                                                                              {kConv2DBackpropFilter, {2}},
                                                                              {kConv2DBackpropInput, {2}},
                                                                              {kOneHot, {1, 3}},
                                                                              {kDropoutGenMask, {0}}};

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (device == kAscendDevice) {
    (void)dynamic_shape_depends.emplace(kReduceSum, std::vector<int64_t>{1});
    (void)dynamic_shape_depends.emplace(kTranspose, std::vector<int64_t>{1});
  }

  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->inputs().empty()) {
    MS_LOG(EXCEPTION) << "Invalid inputs";
  }
  auto primitive = GetValueNode<PrimitivePtr>(cnode->inputs()[0]);
  MS_EXCEPTION_IF_NULL(primitive);
  auto iter = dynamic_shape_depends.find(primitive->ToString());
  if (iter != dynamic_shape_depends.end()) {
    return iter->second;
  }
  return {};
}

PrimitiveEvalImplMap &GetPrimitiveToEvalImplMap() {
  static PrimitiveEvalImplMap prim_eval_implement_map = {
    // Statements
    {prim::kPrimReturn, {InferImplReturn, nullptr, true}},
    {prim::kPrimSwitch, {InferImplSwitch, nullptr, true}},
    {prim::kPrimSwitchLayer, {InferImplSwitchLayer, nullptr, true}},
    {prim::kPrimIs_, {InferImplIs_, nullptr, true}},
    {prim::kPrimIsNot, {InferImplIsNot, nullptr, true}},
    {prim::kPrimInDict, {InferImplInDict, nullptr, true}},
    {prim::kPrimNotInDict, {InferImplNotInDict, nullptr, true}},
    {prim::kPrimIsConsant, {InferImplIsConstant, nullptr, true}},
    // Maths
    {prim::kPrimMatMul, {InferImplMatMul, nullptr, true}},
    {prim::kPrimBatchMatMul, {InferImplBatchMatMul, nullptr, true}},
    {prim::kPrimMaximumGrad, {InferImplMinOrMaxGrad, nullptr, true}},
    {prim::kPrimMinimumGrad, {InferImplMinOrMaxGrad, nullptr, true}},
    {prim::kPrimSqrt, {InferImplSqrt, nullptr, true}},
    // Array
    {prim::kPrimRange, {InferImplRange, nullptr, true}},
    {prim::kPrimScalarToArray, {InferImplScalarToArray, nullptr, true}},
    {prim::kPrimArrayToScalar, {InferImplArrayToScalar, nullptr, true}},
    {prim::kPrimBroadcastShape, {InferImplBroadCastShape, nullptr, true}},
    {prim::kPrimUnique, {InferImplUnique, nullptr, true}},
    {prim::kPrimUniqueGrad, {InferImplUniqueGrad, nullptr, true}},
    {prim::kPrimEmbeddingLookup, {InferImplEmbeddingLookup, nullptr, true}},
    {prim::kPrimSparseGatherV2, {InferImplGatherV2, nullptr, true}},
    {prim::kPrimUnsortedSegmentMax, {InferImplUnsortedSegmentMax, nullptr, true}},
    {prim::kPrimUnsortedSegmentMin, {InferImplUnsortedSegmentMin, nullptr, true}},
    {prim::kPrimScatterAdd, {InferImplScatterAdd, nullptr, true}},
    {prim::kPrimScatterSub, {InferImplScatterSub, nullptr, true}},
    {prim::kPrimSubAndFilter, {InferImplSubAndFilter, nullptr, true}},
    {prim::kPrimScatterUpdate, {InferImplScatterUpdate, nullptr, true}},
    {prim::kPrimMapCacheIdx, {InferImplMapCacheIdx, nullptr, true}},
    {prim::kPrimDynamicAssign, {InferImplDynamicAssign, nullptr, true}},
    {prim::kPrimCacheSwapTable, {InferImplCacheSwapTable, nullptr, true}},
    {prim::kPrimUpdateCache, {InferImplUpdateCache, nullptr, true}},
    {prim::kPrimComputeAccidentalHits, {InferImplComputeAccidentalHits, nullptr, true}},
    {prim::kPrimDynamicStitch, {InferImplDynamicStitch, nullptr, true}},
    {prim::kPrimPadAndShift, {InferImplPadAndShift, nullptr, true}},
    {prim::kPrimDynamicShape, {InferImplDynamicShape, nullptr, true}},
    {prim::kPrimMapUniform, {InferImplMapUniform, nullptr, true}},
    {prim::kPrimSplit, {InferImplSplit, nullptr, true}},
    {prim::kPrimSequenceMask, {InferImplSequenceMask, nullptr, true}},
    {prim::kPrimSort, {InferImplSort, nullptr, true}},
    {prim::kPrimMaskedSelect, {InferImplMaskedSelect, nullptr, true}},
    {prim::kPrimTensorCopySlices, {InferImplTensorCopySlices, nullptr, true}},
    // Structure
    {prim::kPrimMakeTuple, {InferImplMakeTuple, nullptr, true}},
    {prim::kPrimMakeList, {InferImplMakeList, nullptr, true}},
    {prim::kPrimMakeDict, {InferImplMakeDict, nullptr, true}},
    {prim::kPrimMakeSlice, {InferImplMakeSlice, nullptr, true}},
    {prim::kPrimMakeKeywordArg, {InferImplMakeKwarg, nullptr, true}},
    {prim::kPrimExtractKeywordArg, {InferImplExtractKwarg, nullptr, true}},
    {prim::kPrimTupleGetItem, {InferImplTupleGetItem, nullptr, true}},
    {prim::kPrimListGetItem, {InferImplListGetItem, nullptr, true}},
    {prim::kPrimTupleSetItem, {InferImplTupleSetItem, nullptr, true}},
    {prim::kPrimListSetItem, {InferImplListSetItem, nullptr, true}},
    {prim::kPrimDictGetItem, {InferImplDictGetItem, nullptr, true}},
    {prim::kPrimDictSetItem, {InferImplDictSetItem, nullptr, true}},
    {prim::kPrimDictGetKeys, {InferImplDictGetKeys, nullptr, true}},
    {prim::kPrimDictGetValues, {InferImplDictGetValues, nullptr, true}},
    {prim::kPrimListAppend, {InferImplListAppend, nullptr, true}},
    {prim::kPrimTupleLen, {InferImplTupleLen, nullptr, true}},
    {prim::kPrimListLen, {InferImplListLen, nullptr, true}},
    {prim::kPrimArrayLen, {InferImplArrayLen, nullptr, true}},
    // NN
    {prim::kPrimPooling, {InferImplPooling, nullptr, true}},
    {prim::kPrimPoolingGrad, {InferImplPoolingGrad, nullptr, true}},
    {prim::kPrimBatchNorm, {InferImplBatchNorm, nullptr, true}},
    {prim::kPrimConv2D, {InferImplConv2D, nullptr, true}},
    {prim::kPrimBiasAdd, {InferImplBiasAdd, nullptr, true}},
    {prim::kPrimBpropCut, {InferImplBpropCut, nullptr, true}},
    {prim::kPrimDropout, {InferImplDropout, nullptr, true}},
    {prim::kPrimSparseApplyFtrl, {InferImplSparseApplyFtrl, nullptr, true}},
    {prim::kPrimSparseApplyProximalAdagrad, {InferImplSparseApplyProximalAdagrad, nullptr, true}},
    {prim::kPrimSGD, {InferImplSGD, nullptr, true}},
    {prim::kPrimCTCGreedyDecoder, {InferImplCTCGreedyDecoder, nullptr, true}},
    {prim::kPrimHSigmoid, {InferImplHSigmoid, nullptr, true}},
    {prim::kPrimHSigmoidGrad, {InferImplHSigmoidGrad, nullptr, true}},
    // Others
    {prim::kPrimIdentity, {InferImplIdentity, nullptr, true}},
    {prim::kPrimLoad, {InferImplLoad, nullptr, true}},
    // Set impl to null as it will use PartialEvaluator;
    {prim::kPrimPartial, {nullptr, nullptr, true}},
    {prim::kPrimEnvGetItem, {InferImplEnvGetItem, nullptr, true}},
    {prim::kPrimEnvSetItem, {InferImplEnvSetItem, nullptr, true}},
    {prim::kPrimEnvAdd, {InferImplEnvAdd, nullptr, true}},
    {prim::kPrimMakeRefKey, {InferImplMakeRefKey, nullptr, true}},
    {prim::kPrimMakeRef, {InferImplMakeRef, nullptr, true}},
    {prim::kPrimGetRefKey, {InferImplGetRefKey, nullptr, true}},
    {prim::kPrimGetRefValue, {InferImplGetRefValue, nullptr, true}},
    {prim::kPrimStateSetItem, {InferImplStateSetItem, nullptr, true}},
    {prim::kPrimDepend, {InferImplDepend, nullptr, true}},
    {prim::kPrimUpdateState, {InferImplUpdateState, nullptr, true}},
    // Debug
    {prim::kPrimDebug, {InferImplDebug, nullptr, true}},
    // Dynamic shape testing
    {prim::kPrimGpuConvertToDynamicShape, {InferImplGpuConvertToDynamicShape, nullptr, true}},
    // SparseTensor
    {prim::kPrimMakeSparseTensor, {InferImplMakeSparseTensor, nullptr, true}},
    {prim::kPrimSparseTensorGetValues, {InferImplSparseTensorGetValues, nullptr, true}},
    {prim::kPrimSparseTensorGetIndices, {InferImplSparseTensorGetIndices, nullptr, true}},
    {prim::kPrimSparseTensorGetDenseShape, {InferImplSparseTensorGetDenseShape, nullptr, true}},
    // RowTensor
    {prim::kPrimMakeRowTensor, {InferImplMakeRowTensor, nullptr, true}},

    {prim::kPrimRowTensorGetValues, {InferImplRowTensorGetValues, nullptr, true}},
    {prim::kPrimRowTensorGetIndices, {InferImplRowTensorGetIndices, nullptr, true}},
    {prim::kPrimRowTensorGetDenseShape, {InferImplRowTensorGetDenseShape, nullptr, true}},
    {prim::kPrimRowTensorAdd, {InferImplRowTensorAdd, nullptr, false}},
    // Comm Ops
    {prim::kPrimAllSwap, {InferImplAllSwap, nullptr, true}},
    {prim::kPrimMemCpyAsync, {InferImplMemCpyAsync, nullptr, true}},
    {prim::kPrimFusedPushWeight, {nullptr, nullptr, true}},
    {prim::kPrimFusedPullWeight, {nullptr, nullptr, true}},
  };
  return prim_eval_implement_map;
}

PrimitiveEvalImplMap &GetPrimitiveToBackendEvalImplMap() {
  static PrimitiveEvalImplMap prim_backend_eval_implement_map = {
    {prim::kPrimMul, {ops::MulInfer, nullptr, true}},
    {prim::kPrimAdd, {ops::AddInfer, nullptr, false}},
    {prim::kPrimSqrtGrad, {InferImplSqrtGrad, nullptr, true}},
    {prim::kPrimSub, {ops::SubInfer, nullptr, false}},
    {prim::kPrimNeg, {ops::NegInfer, nullptr, false}},
    {prim::kPrimTile, {ops::TileInfer, nullptr, true}},
    {prim::kPrimEqual, {ops::EqualInfer, nullptr, true}},
    {prim::kPrimNotEqual, {ops::NotEqualInfer, nullptr, true}},
    {prim::kPrimLog, {ops::LogInfer, nullptr, true}},
    {prim::kPrimReciprocal, {ops::ReciprocalInfer, nullptr, true}},
    {prim::kPrimReduceSum, {ops::ReduceSumInfer, nullptr, true}},
    {prim::kPrimReduceMean, {InferImplReduceFunc, nullptr, true}},
    {prim::kPrimReduceAll, {InferImplReduceFunc, nullptr, true}},
    {prim::kPrimReduceAny, {InferImplReduceFunc, nullptr, true}},
    {prim::kPrimReduceMax, {InferImplReduceFunc, nullptr, true}},
    {prim::kPrimReduceMin, {InferImplReduceFunc, nullptr, true}},
    {prim::kPrimBiasAddGrad, {InferImplBiasAddGrad, nullptr, true}},
    {prim::kPrimReduceScatter, {InferImplReduceScatter, nullptr, true}},
    {prim::kPrimCast, {InferImplCast, nullptr, true}},
    {prim::kPrimExp, {ops::ExpInfer, nullptr, true}},
    {prim::kPrimExpandDims, {InferImplExpandDims, nullptr, true}},
    {prim::kPrimAllReduce, {InferImplAllReduce, nullptr, true}},
    {prim::kPrimBroadcast, {InferImplBroadcast, nullptr, true}},
    {prim::kPrimAllGather, {InferImplAllGather, nullptr, true}},
    {prim::kPrimMinimum, {InferImplMinimum, nullptr, true}},
    {prim::kPrimDivNoNan, {InferImplDivNoNan, nullptr, true}},
    {prim::kPrimLinSpace, {InferImplLinSpace, nullptr, true}},

    {prim::kPrimLess, {InferImplLess, nullptr, true}},
    {prim::kPrimStack, {InferImplStack, nullptr, true}},
    {prim::kPrimPad, {InferImplPad, nullptr, true}},
    {prim::kPrimUnsortedSegmentSum, {InferImplUnsortedSegmentSum, nullptr, true}},
    {prim::kPrimDiv, {InferImplDiv, nullptr, true}},
    {prim::kPrimRealDiv, {ops::RealDivInfer, nullptr, false}},
    {prim::kPrimTranspose, {InferImplTranspose, nullptr, true}},
    {prim::kPrimStridedSlice, {ops::StridedSliceInfer, nullptr, true}},
    {prim::kPrimReshape, {InferImplReshape, nullptr, true}},
    {prim::kPrimConcat, {InferImplConcat, nullptr, true}},
    {prim::kPrimArgMaxWithValue, {InferImplArgMaxWithValue, nullptr, true}},
    {prim::kPrimFusedSparseAdam, {InferImplFusedSparseAdam, nullptr, true}},
    {prim::kPrimTransData, {InferImplTransData, nullptr, true}},
  };
  return prim_backend_eval_implement_map;
}

StandardPrimitiveImplReg GetPrimitiveInferImpl(const PrimitivePtr &primitive) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto iter = GetPrimitiveToEvalImplMap().find(primitive);
  if (iter == GetPrimitiveToEvalImplMap().end()) {
    return {nullptr, nullptr, false};
  }
  return iter->second;
}

void RegisterStandardPrimitiveImpl(const PrimitivePtr &primitive, const StandardPrimitiveImplReg &impl_reg) {
  auto &prim_eval_map = GetPrimitiveToEvalImplMap();
  prim_eval_map[primitive] = impl_reg;
}
}  // namespace abstract
}  // namespace mindspore
