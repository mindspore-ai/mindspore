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
    {prim::kPrimReturn, {InferImplReturn, true}},
    {prim::kPrimSwitch, {InferImplSwitch, true}},
    {prim::kPrimSwitchLayer, {InferImplSwitchLayer, true}},
    {prim::kPrimIs_, {InferImplIs_, true}},
    {prim::kPrimIsNot, {InferImplIsNot, true}},
    {prim::kPrimInDict, {InferImplInDict, true}},
    {prim::kPrimNotInDict, {InferImplNotInDict, true}},
    {prim::kPrimIsConsant, {InferImplIsConstant, true}},
    // Maths
    {prim::kPrimSquare, {InferImplSquare, true}},
    {prim::kPrimMatMul, {InferImplMatMul, true}},
    {prim::kPrimBatchMatMul, {InferImplBatchMatMul, true}},
    {prim::kPrimMaximumGrad, {InferImplMinOrMaxGrad, true}},
    {prim::kPrimMinimumGrad, {InferImplMinOrMaxGrad, true}},
    {prim::kPrimSqrt, {InferImplSqrt, true}},
    // Array
    {prim::kPrimRange, {InferImplRange, true}},
    {prim::kPrimScalarToArray, {InferImplScalarToArray, true}},
    {prim::kPrimArrayToScalar, {InferImplArrayToScalar, true}},
    {prim::kPrimBroadcastShape, {InferImplBroadCastShape, true}},
    {prim::kPrimUnique, {InferImplUnique, true}},
    {prim::kPrimUniqueGrad, {InferImplUniqueGrad, true}},
    {prim::kPrimGather, {InferImplGatherV2, true}},
    {prim::kPrimEmbeddingLookup, {InferImplEmbeddingLookup, true}},
    {prim::kPrimSparseGatherV2, {InferImplGatherV2, true}},
    {prim::kPrimUnsortedSegmentMax, {InferImplUnsortedSegmentMax, true}},
    {prim::kPrimUnsortedSegmentMin, {InferImplUnsortedSegmentMin, true}},
    {prim::kPrimScatterAdd, {InferImplScatterAdd, true}},
    {prim::kPrimSubAndFilter, {InferImplSubAndFilter, true}},
    {prim::kPrimScatterUpdate, {InferImplScatterUpdate, true}},
    {prim::kPrimMapCacheIdx, {InferImplMapCacheIdx, true}},
    {prim::kPrimDynamicAssign, {InferImplDynamicAssign, true}},
    {prim::kPrimCacheSwapTable, {InferImplCacheSwapTable, true}},
    {prim::kPrimUpdateCache, {InferImplUpdateCache, true}},
    {prim::kPrimComputeAccidentalHits, {InferImplComputeAccidentalHits, true}},
    {prim::kPrimPadAndShift, {InferImplPadAndShift, true}},
    {prim::kPrimDynamicShape, {InferImplDynamicShape, true}},
    {prim::kPrimMapUniform, {InferImplMapUniform, true}},
    {prim::kPrimSplit, {InferImplSplit, true}},
    {prim::kPrimSequenceMask, {InferImplSequenceMask, true}},
    // Structure
    {prim::kPrimMakeTuple, {InferImplMakeTuple, true}},
    {prim::kPrimMakeList, {InferImplMakeList, true}},
    {prim::kPrimMakeDict, {InferImplMakeDict, true}},
    {prim::kPrimMakeSlice, {InferImplMakeSlice, true}},
    {prim::kPrimMakeKeywordArg, {InferImplMakeKwarg, true}},
    {prim::kPrimExtractKeywordArg, {InferImplExtractKwarg, true}},
    {prim::kPrimTupleGetItem, {InferImplTupleGetItem, true}},
    {prim::kPrimListGetItem, {InferImplListGetItem, true}},
    {prim::kPrimTupleSetItem, {InferImplTupleSetItem, true}},
    {prim::kPrimListSetItem, {InferImplListSetItem, true}},
    {prim::kPrimDictGetItem, {InferImplDictGetItem, true}},
    {prim::kPrimDictSetItem, {InferImplDictSetItem, true}},
    {prim::kPrimDictGetKeys, {InferImplDictGetKeys, true}},
    {prim::kPrimDictGetValues, {InferImplDictGetValues, true}},
    {prim::kPrimListAppend, {InferImplListAppend, true}},
    {prim::kPrimTupleLen, {InferImplTupleLen, true}},
    {prim::kPrimListLen, {InferImplListLen, true}},
    {prim::kPrimArrayLen, {InferImplArrayLen, true}},
    // NN
    {prim::kPrimPooling, {InferImplPooling, true}},
    {prim::kPrimPoolingGrad, {InferImplPoolingGrad, true}},
    {prim::kPrimBatchNorm, {InferImplBatchNorm, true}},
    {prim::kPrimReluGrad, {InferImplReluGrad, true}},
    {prim::kPrimConv2D, {InferImplConv2D, true}},
    {prim::kPrimBiasAdd, {InferImplBiasAdd, true}},
    {prim::kPrimRelu, {InferImplRelu, true}},
    {prim::kPrimRelu6, {InferImplRelu, true}},
    {prim::kPrimZerosLike, {InferImplZerosLike, true}},
    {prim::kPrimBpropCut, {InferImplBpropCut, true}},
    {prim::kPrimLayerNorm, {InferImplLayerNorm, true}},
    {prim::kPrimLayerNormGrad, {InferImplLayerNormGrad, true}},
    {prim::kPrimDropout, {InferImplDropout, true}},
    {prim::kPrimDropoutGenMask, {InferImplDropoutGenMask, true}},
    {prim::kPrimSparseApplyFtrl, {InferImplSparseApplyFtrl, true}},
    {prim::kPrimSparseApplyProximalAdagrad, {InferImplSparseApplyProximalAdagrad, true}},
    {prim::kPrimSGD, {InferImplSGD, true}},
    {prim::kPrimCTCGreedyDecoder, {InferImplCTCGreedyDecoder, true}},
    // Others
    {prim::kPrimIdentity, {InferImplIdentity, true}},
    // Set impl to null as it will use PartialEvaluator;
    {prim::kPrimPartial, {nullptr, true}},
    {prim::kPrimEnvGetItem, {InferImplEnvGetItem, true}},
    {prim::kPrimEnvSetItem, {InferImplEnvSetItem, true}},
    {prim::kPrimEnvAdd, {InferImplEnvAdd, true}},
    {prim::kPrimMakeRefKey, {InferImplMakeRefKey, true}},
    {prim::kPrimMakeRef, {InferImplMakeRef, true}},
    {prim::kPrimGetRefKey, {InferImplGetRefKey, true}},
    {prim::kPrimGetRefValue, {InferImplGetRefValue, true}},
    {prim::kPrimStateSetItem, {InferImplStateSetItem, true}},
    {prim::kPrimDepend, {InferImplDepend, true}},
    {prim::kPrimUpdateState, {InferImplUpdateState, true}},
    {prim::kPrimControlDepend, {InferImplControlDepend, true}},
    // Debug
    {prim::kPrimDebug, {InferImplDebug, true}},
    // Dynamic shape testing
    {prim::kPrimGpuConvertToDynamicShape, {InferImplGpuConvertToDynamicShape, true}},
    // SparseTensor
    {prim::kPrimMakeSparseTensor, {InferImplMakeSparseTensor, true}},
    {prim::kPrimSparseTensorGetValues, {InferImplSparseTensorGetValues, true}},
    {prim::kPrimSparseTensorGetIndices, {InferImplSparseTensorGetIndices, true}},
    {prim::kPrimSparseTensorGetDenseShape, {InferImplSparseTensorGetDenseShape, true}},
    // RowTensor
    {prim::kPrimMakeRowTensor, {InferImplMakeRowTensor, true}},
    {prim::kPrimRowTensorGetValues, {InferImplRowTensorGetValues, true}},
    {prim::kPrimRowTensorGetIndices, {InferImplRowTensorGetIndices, true}},
    {prim::kPrimRowTensorGetDenseShape, {InferImplRowTensorGetDenseShape, true}},
    {prim::kPrimRowTensorAdd, {InferImplRowTensorAdd, false}},
    // Comm Ops
    {prim::kPrimAllSwap, {InferImplAllSwap, true}},
    {prim::kPrimMemCpyAsync, {InferImplMemCpyAsync, true}},
  };
  return prim_eval_implement_map;
}

PrimitiveEvalImplMap &GetPrimitiveToBackendEvalImplMap() {
  static PrimitiveEvalImplMap prim_backend_eval_implement_map = {
    {prim::kPrimMul, {InferImplMul, true}},
    {prim::kPrimAdd, {InferImplAdd, true}},
    {prim::kPrimSqrtGrad, {InferImplSqrtGrad, true}},
    {prim::kPrimSub, {InferImplSub, true}},
    {prim::kPrimEqual, {InferImplEqual, true}},
    {prim::kPrimReduceSum, {InferImplReduceFunc, true}},
    {prim::kPrimReduceMean, {InferImplReduceFunc, true}},
    {prim::kPrimReduceAll, {InferImplReduceFunc, true}},
    {prim::kPrimReduceAny, {InferImplReduceFunc, true}},
    {prim::kPrimReduceMax, {InferImplReduceFunc, true}},
    {prim::kPrimReduceMin, {InferImplReduceFunc, true}},
    {prim::kPrimBiasAddGrad, {InferImplBiasAddGrad, true}},
    {prim::kPrimReduceScatter, {InferImplReduceScatter, true}},
    {prim::kPrimCast, {InferImplCast, true}},
    {prim::kPrimExpandDims, {InferImplExpandDims, true}},
    {prim::kPrimAllReduce, {InferImplAllReduce, true}},
    {prim::kPrimBroadcast, {InferImplBroadcast, true}},
    {prim::kPrimAllGather, {InferImplAllGather, true}},
    {prim::kPrimMinimum, {InferImplMinimum, true}},
    {prim::kPrimDivNoNan, {InferImplDivNoNan, true}},
    {prim::kPrimLinSpace, {InferImplLinSpace, true}},
    {prim::kPrimAddN, {InferImplAddN, true}},

    {prim::kPrimLess, {InferImplLess, true}},
    {prim::kPrimStack, {InferImplStack, true}},
    {prim::kPrimPad, {InferImplPad, true}},
    {prim::kPrimUnsortedSegmentSum, {InferImplUnsortedSegmentSum, true}},
    {prim::kPrimDiv, {InferImplDiv, true}},
    {prim::kPrimRealDiv, {InferImplRealDiv, true}},
    {prim::kPrimShape, {InferImplShape, false}},
    {prim::kPrimTranspose, {InferImplTranspose, true}},
    {prim::kPrimReshape, {InferImplReshape, true}},
    {prim::kPrimConcat, {InferImplConcat, true}},
    {prim::kPrimArgMaxWithValue, {InferImplArgMaxWithValue, true}},
    {prim::kPrimFusedSparseAdam, {InferImplFusedSparseAdam, true}},
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
