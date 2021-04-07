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

#include "abstract/abstract_function.h"
#include "abstract/infer_functions.h"

namespace mindspore {
namespace abstract {
std::vector<int64_t> GetDependsFormMap(const CNodePtr &cnode) {
  const auto kUnsortedSegmentSum = prim::kPrimUnsortedSegmentSum->name();
  const auto kUnsortedSegmentMin = prim::kPrimUnsortedSegmentMin->name();
  const auto kUnsortedSegmentMax = prim::kPrimUnsortedSegmentMax->name();
  const auto kGather = prim::kPrimGather->name();
  const auto kGatherV2 = prim::kPrimGatherV2->name();
  const auto kDynamicShape = prim::kPrimDynamicShape->name();
  const auto kRange = prim::kPrimRange->name();
  static std::map<std::string, std::vector<int64_t>> dynamic_shape_depends = {
    {kUnsortedSegmentSum, {2}}, {kUnsortedSegmentMin, {2}}, {kUnsortedSegmentMax, {2}}, {kGather, {2}},
    {kGatherV2, {2}},           {kDynamicShape, {0}},       {kRange, {0, 1, 2}},
  };
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
    {prim::kPrimSquare, {InferImplSquare, nullptr, true}},
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
    {prim::kPrimGather, {InferImplGatherV2, nullptr, true}},
    {prim::kPrimEmbeddingLookup, {InferImplEmbeddingLookup, nullptr, true}},
    {prim::kPrimSparseGatherV2, {InferImplGatherV2, nullptr, true}},
    {prim::kPrimUnsortedSegmentMax, {InferImplUnsortedSegmentMax, nullptr, true}},
    {prim::kPrimUnsortedSegmentMin, {InferImplUnsortedSegmentMin, nullptr, true}},
    {prim::kPrimScatterAdd, {InferImplScatterAdd, nullptr, true}},
    {prim::kPrimSubAndFilter, {InferImplSubAndFilter, nullptr, true}},
    {prim::kPrimScatterUpdate, {InferImplScatterUpdate, nullptr, true}},
    {prim::kPrimMapCacheIdx, {InferImplMapCacheIdx, nullptr, true}},
    {prim::kPrimDynamicAssign, {InferImplDynamicAssign, nullptr, true}},
    {prim::kPrimCacheSwapTable, {InferImplCacheSwapTable, nullptr, true}},
    {prim::kPrimUpdateCache, {InferImplUpdateCache, nullptr, true}},
    {prim::kPrimComputeAccidentalHits, {InferImplComputeAccidentalHits, nullptr, true}},
    {prim::kPrimPadAndShift, {InferImplPadAndShift, nullptr, true}},
    {prim::kPrimDynamicShape, {InferImplDynamicShape, nullptr, true}},
    {prim::kPrimMapUniform, {InferImplMapUniform, nullptr, true}},
    {prim::kPrimSplit, {InferImplSplit, nullptr, true}},
    {prim::kPrimSequenceMask, {InferImplSequenceMask, nullptr, true}},
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
    {prim::kPrimReluGrad, {InferImplReluGrad, nullptr, true}},
    {prim::kPrimConv2D, {InferImplConv2D, nullptr, true}},
    {prim::kPrimBiasAdd, {InferImplBiasAdd, nullptr, true}},
    {prim::kPrimRelu, {InferImplRelu, nullptr, true}},
    {prim::kPrimRelu6, {InferImplRelu, nullptr, true}},
    {prim::kPrimZerosLike, {InferImplZerosLike, nullptr, true}},
    {prim::kPrimBpropCut, {InferImplBpropCut, nullptr, true}},
    {prim::kPrimLayerNorm, {InferImplLayerNorm, nullptr, true}},
    {prim::kPrimLayerNormGrad, {InferImplLayerNormGrad, nullptr, true}},
    {prim::kPrimDropout, {InferImplDropout, nullptr, true}},
    {prim::kPrimDropoutGenMask, {InferImplDropoutGenMask, nullptr, true}},
    {prim::kPrimSparseApplyFtrl, {InferImplSparseApplyFtrl, nullptr, true}},
    {prim::kPrimSparseApplyProximalAdagrad, {InferImplSparseApplyProximalAdagrad, nullptr, true}},
    {prim::kPrimSGD, {InferImplSGD, nullptr, true}},
    {prim::kPrimCTCGreedyDecoder, {InferImplCTCGreedyDecoder, nullptr, true}},
    // Others
    {prim::kPrimIdentity, {InferImplIdentity, nullptr, true}},
    {prim::kPrimLoad, {InferImplLoad, nullptr, true}},
    {prim::kPrimAssign, {InferImplAssign, nullptr, true}},
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
  };
  return prim_eval_implement_map;
}

PrimitiveEvalImplMap &GetPrimitiveToBackendEvalImplMap() {
  static PrimitiveEvalImplMap prim_backend_eval_implement_map = {
    {prim::kPrimMul, {InferImplMul, nullptr, true}},
    {prim::kPrimAdd, {InferImplAdd, nullptr, true}},
    {prim::kPrimSqrtGrad, {InferImplSqrtGrad, nullptr, true}},
    {prim::kPrimSub, {InferImplSub, nullptr, true}},
    {prim::kPrimEqual, {InferImplEqual, nullptr, true}},
    {prim::kPrimReduceSum, {InferImplReduceFunc, nullptr, true}},
    {prim::kPrimReduceMean, {InferImplReduceFunc, nullptr, true}},
    {prim::kPrimReduceAll, {InferImplReduceFunc, nullptr, true}},
    {prim::kPrimReduceAny, {InferImplReduceFunc, nullptr, true}},
    {prim::kPrimReduceMax, {InferImplReduceFunc, nullptr, true}},
    {prim::kPrimReduceMin, {InferImplReduceFunc, nullptr, true}},
    {prim::kPrimBiasAddGrad, {InferImplBiasAddGrad, nullptr, true}},
    {prim::kPrimReduceScatter, {InferImplReduceScatter, nullptr, true}},
    {prim::kPrimCast, {InferImplCast, nullptr, true}},
    {prim::kPrimExpandDims, {InferImplExpandDims, nullptr, true}},
    {prim::kPrimAllReduce, {InferImplAllReduce, nullptr, true}},
    {prim::kPrimBroadcast, {InferImplBroadcast, nullptr, true}},
    {prim::kPrimAllGather, {InferImplAllGather, nullptr, true}},
    {prim::kPrimMinimum, {InferImplMinimum, nullptr, true}},
    {prim::kPrimDivNoNan, {InferImplDivNoNan, nullptr, true}},
    {prim::kPrimLinSpace, {InferImplLinSpace, nullptr, true}},
    {prim::kPrimAddN, {InferImplAddN, nullptr, true}},

    {prim::kPrimLess, {InferImplLess, nullptr, true}},
    {prim::kPrimStack, {InferImplStack, nullptr, true}},
    {prim::kPrimPad, {InferImplPad, nullptr, true}},
    {prim::kPrimUnsortedSegmentSum, {InferImplUnsortedSegmentSum, nullptr, true}},
    {prim::kPrimDiv, {InferImplDiv, nullptr, true}},
    {prim::kPrimRealDiv, {InferImplRealDiv, nullptr, true}},
    {prim::kPrimShape, {InferImplShape, nullptr, false}},
    {prim::kPrimTranspose, {InferImplTranspose, nullptr, true}},
    {prim::kPrimReshape, {InferImplReshape, nullptr, true}},
    {prim::kPrimConcat, {InferImplConcat, nullptr, true}},
    {prim::kPrimArgMaxWithValue, {InferImplArgMaxWithValue, nullptr, true}},
    {prim::kPrimFusedSparseAdam, {InferImplFusedSparseAdam, nullptr, true}},
  };
  return prim_backend_eval_implement_map;
}

StandardPrimitiveEvalImpl GetPrimitiveInferImpl(const PrimitivePtr &primitive) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto iter = GetPrimitiveToEvalImplMap().find(primitive);
  if (iter == GetPrimitiveToEvalImplMap().end()) {
    return nullptr;
  }
  return iter->second.impl_;
}

void RegisterStandardPrimitiveImpl(const PrimitivePtr &primitive, const StandardPrimitiveImplReg &impl_reg) {
  auto &prim_eval_map = GetPrimitiveToEvalImplMap();
  prim_eval_map[primitive] = impl_reg;
}
}  // namespace abstract
}  // namespace mindspore
