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
#include "abstract/abstract_function.h"
#include "abstract/infer_functions.h"

namespace mindspore {
namespace abstract {
PrimitiveEvalImplMap &GetPrimitiveToEvalImplMap() {
  static PrimitiveEvalImplMap prim_eval_implement_map = {
    // Statements
    {prim::kPrimReturn, {InferImplReturn, true}},
    {prim::kPrimDot, {InferImplDot, true}},
    {prim::kPrimSwitch, {InferImplSwitch, true}},
    {prim::kPrimSwitchLayer, {InferImplSwitchLayer, true}},
    {prim::kPrimIs_, {InferImplIs_, true}},
    {prim::kPrimIsNot, {InferImplIsNot, true}},
    {prim::kPrimInDict, {InferImplInDict, true}},
    {prim::kPrimNotInDict, {InferImplNotInDict, true}},
    {prim::kPrimIsConsant, {InferImplIsConstant, true}},
    // Maths
    {prim::kPrimMaximumGrad, {InferImplMinOrMaxGrad, true}},
    {prim::kPrimMinimumGrad, {InferImplMinOrMaxGrad, true}},
    {prim::kPrimMul, {InferImplMul, true}},
    {prim::kPrimTensorAdd, {InferImplTensorAdd, true}},
    {prim::kPrimSquare, {InferImplSquare, true}},
    {prim::kPrimSqrt, {InferImplSqrt, true}},
    {prim::kPrimSqrtGrad, {InferImplSqrtGrad, true}},
    {prim::kPrimSub, {InferImplSub, true}},
    {prim::kPrimEqual, {InferImplEqual, true}},
    {prim::kPrimReduceSum, {InferImplReduceFunc, true}},
    {prim::kPrimReduceMean, {InferImplReduceFunc, true}},
    {prim::kPrimReduceAll, {InferImplReduceFunc, true}},
    {prim::kPrimReduceAny, {InferImplReduceFunc, true}},
    {prim::kPrimReduceMax, {InferImplReduceFunc, true}},
    {prim::kPrimReduceMin, {InferImplReduceFunc, true}},
    {prim::kPrimMinimum, {InferImplMinimum, true}},
    {prim::kPrimDivNoNan, {InferImplDivNoNan, true}},
    {prim::kPrimLinSpace, {InferImplLinSpace, true}},
    {prim::kPrimAddN, {InferImplAddN, true}},
    {prim::kPrimMatMul, {InferImplMatMul, true}},
    {prim::kPrimBatchMatMul, {InferImplBatchMatMul, true}},
    // Array
    {prim::kPrimScalarToArray, {InferImplScalarToArray, true}},
    {prim::kPrimArrayToScalar, {InferImplArrayToScalar, true}},
    {prim::kPrimBroadcastShape, {InferImplBroadCastShape, true}},
    {prim::kPrimPack, {InferImplPack, true}},
    {prim::kPrimPad, {InferImplPad, true}},
    {prim::kPrimUnique, {InferImplUnique, true}},
    {prim::kPrimUniqueGrad, {InferImplUniqueGrad, true}},
    {prim::kPrimGatherV2, {InferImplGatherV2, true}},
    {prim::kPrimSparseGatherV2, {InferImplGatherV2, true}},
    {prim::kPrimEmbeddingLookup, {InferImplEmbeddingLookup, true}},
    {prim::kPrimUnsortedSegmentSum, {InferImplUnsortedSegmentSum, true}},
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
    {prim::kPrimDiv, {InferImplDiv, true}},
    {prim::kPrimRealDiv, {InferImplRealDiv, true}},
    {prim::kPrimShape, {InferImplShape, false}},
    {prim::kPrimDynamicShape, {InferImplDynamicShape, true}},
    {prim::kPrimTranspose, {InferImplTranspose, true}},
    {prim::kPrimReshape, {InferImplReshape, true}},
    {prim::kPrimMapUniform, {InferImplMapUniform, true}},
    {prim::kPrimSplit, {InferImplSplit, true}},
    {prim::kPrimSequenceMask, {InferImplSequenceMask, true}},
    {prim::kPrimConcat, {InferImplConcat, true}},
    {prim::kPrimRange, {InferImplRange, true}},
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
    {prim::kPrimFusedBatchNorm, {InferImplFusedBatchNorm, true}},
    {prim::kPrimFusedSparseAdam, {InferImplFusedSparseAdam, true}},
    {prim::kPrimFusedBatchNormGrad, {InferImplFusedBatchNormGrad, true}},
    {prim::kPrimBatchNormGrad, {InferImplBatchNormGrad, true}},
    {prim::kPrimReluGrad, {InferImplReluGrad, true}},
    {prim::kPrimConv2D, {InferImplConv2D, true}},
    {prim::kPrimConv2DBackpropInput, {InferImplConv2DBackpropInput, true}},
    {prim::kPrimConv2DBackpropFilter, {InferImplConv2DBackpropFilter, true}},
    {prim::kPrimBiasAdd, {InferImplBiasAdd, true}},
    {prim::kPrimBiasAddGrad, {InferImplBiasAddGrad, true}},
    {prim::kPrimRelu, {InferImplRelu, true}},
    {prim::kPrimZerosLike, {InferImplZerosLike, true}},
    {prim::kPrimBpropCut, {InferImplBpropCut, true}},
    {prim::kPrimLayerNorm, {InferImplLayerNorm, true}},
    {prim::kPrimLayerNormGrad, {InferImplLayerNormGrad, true}},
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
    // Comm Ops
    {prim::kPrimAllReduce, {InferImplAllReduce, true}},
    {prim::kPrimBroadcast, {InferImplBroadcast, true}},
    {prim::kPrimAllGather, {InferImplAllGather, true}},
    {prim::kPrimAllSwap, {InferImplAllSwap, true}},
    {prim::kPrimReduceScatter, {InferImplReduceScatter, true}},
    {prim::kPrimMemCpyAsync, {InferImplMemCpyAsync, true}},
    {prim::kPrimCast, {InferImplCast, true}},
    {prim::kPrimExpandDims, {InferImplExpandDims, true}},
  };
  return prim_eval_implement_map;
}

void RegisterStandardPrimitiveImpl(const PrimitivePtr &primitive, const StandardPrimitiveImplReg &impl_reg) {
  auto &prim_eval_map = GetPrimitiveToEvalImplMap();
  prim_eval_map[primitive] = impl_reg;
}
}  // namespace abstract
}  // namespace mindspore
