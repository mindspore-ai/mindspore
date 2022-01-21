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
#include <string>
#include <vector>
#include <set>
#include "ops/exp.h"
#include "ops/log.h"
#include "ops/reciprocal.h"
#include "ops/real_div.h"
#include "ops/add.h"
#include "ops/equal.h"
#include "ops/greater_equal.h"
#include "ops/greater.h"
#include "ops/not_equal.h"
#include "ops/neg.h"
#include "ops/mul.h"
#include "ops/mod.h"
#include "ops/sub.h"
#include "ops/strided_slice.h"
#include "ops/reduce_sum.h"
#include "abstract/abstract_function.h"
#include "abstract/infer_functions.h"
#include "utils/ms_context.h"
#include "ops/tile.h"
#include "ops/slice.h"
#include "ops/grad/slice_grad.h"
#include "ops/lstm.h"

namespace mindspore {
namespace abstract {
std::set<int64_t> GetDependsFormMap(const CNodePtr &cnode) {
  using ShapeSet = std::set<int64_t>;
  using PrimShapeDependMap = mindspore::HashMap<std::string, ShapeSet>;
  static const auto &kOneHot = prim::kPrimOneHot->name();
  static const auto &kDropoutGenMask = prim::kPrimDropoutGenMask->name();
  static const auto &kTranspose = prim::kPrimTranspose->name();
  static const auto &kStridedSlice = prim::kPrimStridedSlice->name();
  static const auto &kStridedSliceGrad = prim::kPrimStridedSliceGrad->name();
  static const auto &kReduceSum = prim::kPrimReduceSum->name();
  static const auto &kDynamicBroadcastTo = prim::kPrimDynamicBroadcastTo->name();
  static const auto &kUnsortedSegmentSum = prim::kPrimUnsortedSegmentSum->name();
  static const auto &kUnsortedSegmentMin = prim::kPrimUnsortedSegmentMin->name();
  static const auto &kUnsortedSegmentMax = prim::kPrimUnsortedSegmentMax->name();
  static const auto &kGather = prim::kPrimGather->name();
  static const auto &kGatherV2 = prim::kPrimGatherV2->name();
  static const auto &kRange = prim::kPrimRange->name();
  static const auto &kConv2DBackpropFilter = prim::kPrimConv2DBackpropFilter->name();
  static const auto &kConv2DBackpropInput = prim::kPrimConv2DBackpropInput->name();
  static const auto &kTile = prim::kPrimTile->name();
  static const auto &kSlice = prim::kPrimSlice->name();
  static const auto &kSliceGrad = prim::kPrimSliceGrad->name();
  static const auto &kReshape = prim::kPrimReshape->name();
  // Common dynamic shape depends.
  static const PrimShapeDependMap dynamic_shape_depends{{kUnsortedSegmentSum, ShapeSet{2}},
                                                        {kUnsortedSegmentMin, ShapeSet{2}},
                                                        {kUnsortedSegmentMax, ShapeSet{2}},
                                                        {kGather, ShapeSet{2}},
                                                        {kGatherV2, ShapeSet{2}},
                                                        {kRange, ShapeSet{0, 1, 2}},
                                                        {kConv2DBackpropFilter, ShapeSet{2}},
                                                        {kConv2DBackpropInput, ShapeSet{2}},
                                                        {kOneHot, ShapeSet{1, 3}},
                                                        {kDropoutGenMask, ShapeSet{0}},
                                                        {kStridedSlice, ShapeSet{1, 2, 3}},
                                                        {kStridedSliceGrad, ShapeSet{1, 2, 3, 4}},
                                                        {kTile, ShapeSet{1}},
                                                        {kReshape, ShapeSet{1}},
                                                        {kSlice, ShapeSet{1, 2}},
                                                        {kSliceGrad, ShapeSet{2, 3}},
                                                        {kDynamicBroadcastTo, ShapeSet{1}}};

  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->inputs().empty()) {
    MS_LOG(EXCEPTION) << "Invalid inputs";
  }
  auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->ToString();

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  // Special dynamic shape depends for Ascend.
  if (device == kAscendDevice && (prim_name == kReduceSum || prim_name == kTranspose)) {
    return {1};
  }

  auto iter = dynamic_shape_depends.find(prim_name);
  if (iter != dynamic_shape_depends.end()) {
    int64_t cnode_input_size = SizeToLong(cnode->inputs().size());
    ShapeSet res;
    auto ori = iter->second;
    (void)std::copy_if(ori.begin(), ori.end(), std::inserter(res, res.begin()),
                       [&](auto idx) { return idx < cnode_input_size - 1; });
    return res;
  }
  return {};
}

PrimitiveEvalImplMap &GetPrimitiveToEvalImplMap() {
  using R = PrimitiveEvalImplMap::mapped_type;
  static PrimitiveEvalImplMap prim_eval_implement_map{
    // Statements
    {prim::kPrimReturn, R{InferImplReturn, nullptr, true}},
    {prim::kPrimSwitch, R{InferImplSwitch, nullptr, true}},
    {prim::kPrimSwitchLayer, R{InferImplSwitchLayer, nullptr, true}},
    {prim::kPrimIs_, R{InferImplIs_, nullptr, true}},
    {prim::kPrimIsNot, R{InferImplIsNot, nullptr, true}},
    {prim::kPrimInDict, R{InferImplInDict, nullptr, true}},
    {prim::kPrimNotInDict, R{InferImplNotInDict, nullptr, true}},
    {prim::kPrimIsConsant, R{InferImplIsConstant, nullptr, true}},
    // Maths
    {prim::kPrimMatMul, R{InferImplMatMul, nullptr, true}},
    {prim::kPrimBatchMatMul, R{InferImplBatchMatMul, nullptr, true}},
    {prim::kPrimMaximumGrad, R{InferImplMinOrMaxGrad, nullptr, true}},
    {prim::kPrimMinimumGrad, R{InferImplMinOrMaxGrad, nullptr, true}},
    {prim::kPrimSqrt, R{InferImplSqrt, nullptr, true}},
    {prim::kPrimRealInner, R{InferImplReal, nullptr, true}},
    // Array
    {prim::kPrimRange, R{InferImplRange, nullptr, true}},
    {prim::kPrimScalarToArray, R{InferImplScalarToArray, nullptr, true}},
    {prim::kPrimArrayToScalar, R{InferImplArrayToScalar, nullptr, true}},
    {prim::kPrimBroadcastShape, R{InferImplBroadCastShape, nullptr, true}},
    {prim::kPrimUnique, R{InferImplUnique, nullptr, true}},
    {prim::kPrimUniqueGrad, R{InferImplUniqueGrad, nullptr, true}},
    {prim::kPrimEmbeddingLookup, R{InferImplEmbeddingLookup, nullptr, true}},
    {prim::kPrimSparseGatherV2, R{InferImplGatherV2, nullptr, true}},
    {prim::kPrimUnsortedSegmentMax, R{InferImplUnsortedSegmentMax, nullptr, true}},
    {prim::kPrimUnsortedSegmentMin, R{InferImplUnsortedSegmentMin, nullptr, true}},
    {prim::kPrimScatterAdd, R{InferImplScatterAdd, nullptr, true}},
    {prim::kPrimScatterSub, R{InferImplScatterSub, nullptr, true}},
    {prim::kPrimScatterElements, R{InferImplScatterElements, nullptr, true}},
    {prim::kPrimSubAndFilter, R{InferImplSubAndFilter, nullptr, true}},
    {prim::kPrimScatterUpdate, R{InferImplScatterUpdate, nullptr, true}},
    {prim::kPrimMapCacheIdx, R{InferImplMapCacheIdx, nullptr, true}},
    {prim::kPrimDynamicAssign, R{InferImplDynamicAssign, nullptr, true}},
    {prim::kPrimCacheSwapTable, R{InferImplCacheSwapTable, nullptr, true}},
    {prim::kPrimUpdateCache, R{InferImplUpdateCache, nullptr, true}},
    {prim::kPrimComputeAccidentalHits, R{InferImplComputeAccidentalHits, nullptr, true}},
    {prim::kPrimDynamicStitch, R{InferImplDynamicStitch, nullptr, true}},
    {prim::kPrimPadAndShift, R{InferImplPadAndShift, nullptr, true}},
    {prim::kPrimDynamicShape, R{InferImplDynamicShape, nullptr, true}},
    {prim::kPrimMapUniform, R{InferImplMapUniform, nullptr, true}},
    {prim::kPrimSplit, R{InferImplSplit, nullptr, true}},
    {prim::kPrimSequenceMask, R{InferImplSequenceMask, nullptr, true}},
    {prim::kPrimSort, R{InferImplSort, nullptr, true}},
    {prim::kPrimMaskedSelect, R{InferImplMaskedSelect, nullptr, true}},
    {prim::kPrimTensorCopySlices, R{InferImplTensorCopySlices, nullptr, true}},
    {prim::kPrimNonZero, R{InferImplNonZero, nullptr, true}},
    // Structure
    {prim::kPrimMakeTuple, R{InferImplMakeTuple, nullptr, true}},
    {prim::kPrimMakeList, R{InferImplMakeList, nullptr, true}},
    {prim::kPrimMakeDict, R{InferImplMakeDict, nullptr, true}},
    {prim::kPrimMakeKeywordArg, R{InferImplMakeKwarg, nullptr, true}},
    {prim::kPrimExtractKeywordArg, R{InferImplExtractKwarg, nullptr, true}},
    {prim::kPrimTupleGetItem, R{InferImplTupleGetItem, nullptr, true}},
    {prim::kPrimListGetItem, R{InferImplListGetItem, nullptr, true}},
    {prim::kPrimTupleSetItem, R{InferImplTupleSetItem, nullptr, true}},
    {prim::kPrimListSetItem, R{InferImplListSetItem, nullptr, true}},
    {prim::kPrimDictGetItem, R{InferImplDictGetItem, nullptr, true}},
    {prim::kPrimDictSetItem, R{InferImplDictSetItem, nullptr, true}},
    {prim::kPrimDictGetKeys, R{InferImplDictGetKeys, nullptr, true}},
    {prim::kPrimDictGetValues, R{InferImplDictGetValues, nullptr, true}},
    {prim::kPrimDictItems, R{InferImplDictItems, nullptr, true}},
    {prim::kPrimListAppend, R{InferImplListAppend, nullptr, true}},
    {prim::kPrimTupleLen, R{InferImplTupleLen, nullptr, true}},
    {prim::kPrimListLen, R{InferImplListLen, nullptr, true}},
    {prim::kPrimArrayLen, R{InferImplArrayLen, nullptr, true}},
    // NN
    {prim::kPrimPooling, R{InferImplPooling, nullptr, true}},
    {prim::kPrimPoolingGrad, R{InferImplPoolingGrad, nullptr, true}},
    {prim::kPrimBatchNorm, R{InferImplBatchNorm, nullptr, true}},
    {prim::kPrimBpropCut, R{InferImplBpropCut, nullptr, true}},
    {prim::kPrimDropout, R{InferImplDropout, nullptr, true}},
    {prim::kPrimSparseApplyFtrl, R{InferImplSparseApplyFtrl, nullptr, true}},
    {prim::kPrimSparseApplyProximalAdagrad, R{InferImplSparseApplyProximalAdagrad, nullptr, true}},
    {prim::kPrimSGD, R{InferImplSGD, nullptr, true}},
    {prim::kPrimCTCGreedyDecoder, R{InferImplCTCGreedyDecoder, nullptr, true}},
    {prim::kPrimHSigmoid, R{InferImplHSigmoid, nullptr, true}},
    {prim::kPrimHSigmoidGrad, R{InferImplHSigmoidGrad, nullptr, true}},
    // Others
    {prim::kPrimIdentity, R{InferImplIdentity, nullptr, true}},
    {prim::kPrimLoad, R{InferImplLoad, nullptr, true}},
    // Set impl to null as it will use PartialEvaluator;
    {prim::kPrimPartial, R{nullptr, nullptr, true}},
    {prim::kPrimEnvironCreate, R{InferImplEnvironCreate, nullptr, true}},
    {prim::kPrimEnvironGet, R{InferImplEnvironGet, nullptr, true}},
    {prim::kPrimEnvironSet, R{InferImplEnvironSet, nullptr, true}},
    {prim::kPrimEnvironAdd, R{InferImplEnvironAdd, nullptr, true}},
    {prim::kPrimMakeRefKey, R{InferImplMakeRefKey, nullptr, true}},
    {prim::kPrimMakeRef, R{InferImplMakeRef, nullptr, true}},
    {prim::kPrimGetRefKey, R{InferImplGetRefKey, nullptr, true}},
    {prim::kPrimGetRefValue, R{InferImplGetRefValue, nullptr, true}},
    {prim::kPrimStateSetItem, R{InferImplStateSetItem, nullptr, true}},
    {prim::kPrimDepend, R{InferImplDepend, nullptr, true}},
    {prim::kPrimUpdateState, R{InferImplUpdateState, nullptr, true}},
    // Debug
    {prim::kPrimDebug, R{InferImplDebug, nullptr, true}},
    // Dynamic shape testing
    {prim::kPrimGpuConvertToDynamicShape, R{InferImplGpuConvertToDynamicShape, nullptr, true}},
    // SparseTensor
    {prim::kPrimMakeSparseTensor, R{InferImplMakeSparseTensor, nullptr, true}},
    {prim::kPrimSparseTensorGetValues, R{InferImplSparseTensorGetValues, nullptr, true}},
    {prim::kPrimSparseTensorGetIndices, R{InferImplSparseTensorGetIndices, nullptr, true}},
    {prim::kPrimSparseTensorGetDenseShape, R{InferImplSparseTensorGetDenseShape, nullptr, true}},
    // RowTensor
    {prim::kPrimMakeRowTensor, R{InferImplMakeRowTensor, nullptr, true}},
    {prim::kPrimRowTensorGetValues, R{InferImplRowTensorGetValues, nullptr, true}},
    {prim::kPrimRowTensorGetIndices, R{InferImplRowTensorGetIndices, nullptr, true}},
    {prim::kPrimRowTensorGetDenseShape, R{InferImplRowTensorGetDenseShape, nullptr, true}},
    {prim::kPrimRowTensorAdd, R{InferImplRowTensorAdd, nullptr, false}},
    // CSRTensor
    {prim::kPrimMakeCSRTensor, R{InferImplMakeCSRTensor, nullptr, true}},
    {prim::kPrimCSRTensorGetValues, R{InferImplCSRTensorGetValues, nullptr, true}},
    {prim::kPrimCSRTensorGetIndptr, R{InferImplCSRTensorGetIndptr, nullptr, true}},
    {prim::kPrimCSRTensorGetIndices, R{InferImplCSRTensorGetIndices, nullptr, true}},
    {prim::kPrimCSRTensorGetDenseShape, R{InferImplCSRTensorGetDenseShape, nullptr, true}},
    {prim::kPrimCSRMul, R{InferImplCSRMul, nullptr, true}},
    {prim::kPrimCSRMV, R{InferImplCSRMV, nullptr, true}},
    {prim::kPrimCSRReduceSum, R{InferImplCSRReduceSum, nullptr, true}},
    // Comm Ops
    {prim::kPrimAllSwap, R{InferImplAllSwap, nullptr, true}},
    {prim::kPrimMemCpyAsync, R{InferImplMemCpyAsync, nullptr, true}},
    {prim::kPrimFusedPushWeight, R{nullptr, nullptr, true}},
    {prim::kPrimFusedPullWeight, R{nullptr, nullptr, true}},
    // RL Ops
    {prim::kPrimTensorArrayStack, R{InferImplTensorArrayStack, nullptr, true}},
  };
  return prim_eval_implement_map;
}

PrimitiveEvalImplMap &GetPrimitiveToBackendEvalImplMap() {
  using R = PrimitiveEvalImplMap::mapped_type;
  static PrimitiveEvalImplMap prim_backend_eval_implement_map = {
    {prim::kPrimMul, R{ops::MulInfer, nullptr, true}},
    {prim::kPrimMod, R{ops::ModInfer, nullptr, true}},
    {prim::kPrimAdd, R{ops::AddInfer, nullptr, false}},
    {prim::kPrimSqrtGrad, R{InferImplSqrtGrad, nullptr, true}},
    {prim::kPrimSub, R{ops::SubInfer, nullptr, false}},
    {prim::kPrimNeg, R{ops::NegInfer, nullptr, false}},
    {prim::kPrimTile, R{ops::TileInfer, nullptr, true}},
    {prim::kPrimEqual, R{ops::EqualInfer, nullptr, true}},
    {prim::kPrimGreater, R{ops::GreaterInfer, nullptr, true}},
    {prim::kPrimGreaterEqual, R{ops::GreaterEqualInfer, nullptr, true}},
    {prim::kPrimNotEqual, R{ops::NotEqualInfer, nullptr, true}},
    {prim::kPrimLog, R{ops::LogInfer, nullptr, true}},
    {prim::kPrimReciprocal, R{ops::ReciprocalInfer, nullptr, true}},
    {prim::kPrimReduceSum, R{ops::ReduceSumInfer, nullptr, true}},
    {prim::kPrimReduceMean, R{InferImplReduceFunc, nullptr, true}},
    {prim::kPrimReduceAll, R{InferImplReduceFunc, nullptr, true}},
    {prim::kPrimReduceAny, R{InferImplReduceFunc, nullptr, true}},
    {prim::kPrimReduceMax, R{InferImplReduceFunc, nullptr, true}},
    {prim::kPrimReduceMin, R{InferImplReduceFunc, nullptr, true}},
    {prim::kPrimBiasAddGrad, R{InferImplBiasAddGrad, nullptr, true}},
    {prim::kPrimReduceScatter, R{InferImplReduceScatter, nullptr, true}},
    {prim::kPrimCast, R{InferImplCast, nullptr, true}},
    {prim::kPrimExp, R{ops::ExpInfer, nullptr, true}},
    {prim::kPrimExpandDims, R{InferImplExpandDims, nullptr, true}},
    {prim::kPrimAllReduce, R{InferImplAllReduce, nullptr, true}},
    {prim::kPrimBroadcast, R{InferImplBroadcast, nullptr, true}},
    {prim::kPrimAllGather, R{InferImplAllGather, nullptr, true}},
    {prim::kPrimMinimum, R{InferImplMinimum, nullptr, true}},
    {prim::kPrimDivNoNan, R{InferImplDivNoNan, nullptr, true}},
    {prim::kPrimLinSpace, R{InferImplLinSpace, nullptr, true}},

    {prim::kPrimLess, R{InferImplLess, nullptr, true}},
    {prim::kPrimStack, R{InferImplStack, nullptr, true}},
    {prim::kPrimPad, R{InferImplPad, nullptr, true}},
    {prim::kPrimUnsortedSegmentSum, R{InferImplUnsortedSegmentSum, nullptr, true}},
    {prim::kPrimDiv, R{InferImplDiv, nullptr, true}},
    {prim::kPrimRealDiv, R{ops::RealDivInfer, nullptr, false}},
    {prim::kPrimTranspose, R{InferImplTranspose, nullptr, true}},
    {prim::kPrimStridedSlice, R{ops::StridedSliceInfer, nullptr, true}},
    {prim::kPrimSlice, R{ops::SliceInfer, nullptr, true}},
    {prim::kPrimSliceGrad, R{ops::SliceGradInfer, nullptr, true}},
    {prim::kPrimReshape, R{InferImplReshape, nullptr, true}},
    {prim::kPrimConcat, R{InferImplConcat, nullptr, true}},
    {prim::kPrimConcatOffset, R{InferImplConcatOffset, nullptr, true}},
    {prim::kPrimArgMaxWithValue, R{InferImplArgMaxWithValue, nullptr, true}},
    {prim::kPrimFusedSparseAdam, R{InferImplFusedSparseAdam, nullptr, true}},
    {prim::kPrimTransData, R{InferImplTransData, nullptr, true}},
    {prim::kPrimLstm, R{ops::LstmInfer, nullptr, true}},
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
