/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "abstract/ops/primitive_infer_map.h"
#include <string>
#include <vector>
#include <set>
#include "ops/exp.h"
#include "ops/log.h"
#include "ops/reciprocal.h"
#include "ops/real_div.h"
#include "ops/add.h"
#include "ops/arg_min.h"
#include "ops/equal.h"
#include "ops/greater_equal.h"
#include "ops/greater.h"
#include "ops/not_equal.h"
#include "ops/neg.h"
#include "ops/mul.h"
#include "ops/mod.h"
#include "ops/sub.h"
#include "ops/strided_slice.h"
#include "ops/strided_slice_v2.h"
#include "ops/grad/strided_slice_v2_grad.h"
#include "abstract/abstract_function.h"
#include "abstract/ops/infer_functions.h"
#include "utils/ms_context.h"
#include "ops/tile.h"
#include "ops/slice.h"
#include "ops/grad/slice_grad.h"
#include "ops/lstm.h"
#include "ops/stack.h"
#include "ops/rpc_recv.h"
#include "ops/rpc_send.h"
#include "ops/tensor_scatter_arithmetic.h"
#include "ops/max_pool.h"
#include "ops/grad/max_pool_grad.h"
#include "ops/dropout.h"
#include "ops/extract_image_patches.h"
#include "ops/grad/max_pool_grad_with_argmax.h"
#include "ops/max_pool_with_argmax.h"
#include "ops/mirror_pad.h"
#include "ops/opaquePredicate.h"

namespace mindspore {
namespace abstract {
using ops::InferImplDropout;

PrimShapeDependMap &GetHostDependsMap() {
  // Registration directly by the host_depends map will be deprecated and
  // should be registered by the REGISTER_HOST_DEPENDS
  using ShapeSet = std::set<int64_t>;
  static const auto &kMirrorPad = prim::kPrimMirrorPad->name();
  static const auto &kAdaptiveMaxPool3DGrad = prim::kPrimAdaptiveMaxPool3DGrad->name();
  static const auto &kAdaptiveAvgPool3D = prim::kPrimAdaptiveAvgPool3D->name();
  static const auto &kAdaptiveAvgPool3DGrad = prim::kPrimAdaptiveAvgPool3DGrad->name();
  static const auto &kOneHot = prim::kPrimOneHot->name();
  static const auto &kDropoutGenMask = prim::kPrimDropoutGenMask->name();
  static const auto &kStridedSlice = prim::kPrimStridedSlice->name();
  static const auto &kStridedSliceGrad = prim::kPrimStridedSliceGrad->name();
  static const auto &kStridedSliceV2 = prim::kPrimStridedSliceV2->name();
  static const auto &kStridedSliceV2Grad = prim::kPrimStridedSliceV2Grad->name();
  static const auto &kSparseToDenseV2 = prim::kPrimSparseToDenseV2->name();
  static const auto &kResizeBicubic = prim::kPrimResizeBicubic->name();
  static const auto &kRandomCategorical = prim::kPrimRandomCategorical->name();
  static const auto &kMatrixDiagV3 = prim::kPrimMatrixDiagV3->name();
  static const auto &kMatrixDiagPartV3 = prim::kPrimMatrixDiagPartV3->name();
  static const auto &kMatrixSetDiagV3 = prim::kPrimMatrixSetDiagV3->name();
  static const auto &kRaggedRange = prim::kPrimRaggedRange->name();
  static const auto &kDynamicBroadcastTo = prim::kPrimDynamicBroadcastTo->name();
  static const auto &kUnsortedSegmentSum = prim::kPrimUnsortedSegmentSum->name();
  static const auto &kUnsortedSegmentProd = prim::kPrimUnsortedSegmentProd->name();
  static const auto &kUnsortedSegmentMin = prim::kPrimUnsortedSegmentMin->name();
  static const auto &kUnsortedSegmentMax = prim::kPrimUnsortedSegmentMax->name();
  static const auto &kGather = prim::kPrimGather->name();
  static const auto &kGatherV2 = prim::kPrimGatherV2->name();
  static const auto &kGatherD = prim::kPrimGatherD->name();
  static const auto &kSparseGatherV2 = prim::kPrimSparseGatherV2->name();
  static const auto &kRangeV2 = prim::kPrimRangeV2->name();
  static const auto &kConv2DBackpropFilter = prim::kPrimConv2DBackpropFilter->name();
  static const auto &kConv2DBackpropInput = prim::kPrimConv2DBackpropInput->name();
  static const auto &kCol2Im = prim::kPrimCol2Im->name();
  static const auto &kTile = prim::kPrimTile->name();
  static const auto &kTopK = prim::kPrimTopK->name();
  static const auto &kNonDeterministicInts = prim::kPrimNonDeterministicInts->name();
  static const auto &kSliceGrad = prim::kPrimSliceGrad->name();
  static const auto &kReshape = prim::kPrimReshape->name();
  static const auto &kResizeNearestNeighborV2 = prim::kPrimResizeNearestNeighborV2->name();
  static const auto &kResizeNearestNeighborV2Grad = prim::kPrimResizeNearestNeighborV2Grad->name();
  static const auto &kScatterNd = prim::kPrimScatterNd->name();
  static const auto &kTruncatedNormal = prim::kPrimTruncatedNormal->name();
  static const auto &kRandomGamma = prim::kPrimRandomGamma->name();
  static const auto &kAffineGrid = prim::kPrimAffineGrid->name();
  static const auto &kFillV2 = prim::kPrimFillV2->name();
  static const auto &kFractionalAvgPoolGrad = prim::kPrimFractionalAvgPoolGrad->name();
  static const auto &kTransposeNOD = prim::kPrimTransposeNOD->name();
  static const auto &kResizeLinear1D = prim::kPrimResizeLinear1D->name();
  static const auto &kSegmentMax = prim::kPrimSegmentMax->name();
  static const auto &kSegmentMin = prim::kPrimSegmentMin->name();
  static const auto &kSegmentSum = prim::kPrimSegmentSum->name();
  static const auto &kBlackmanWindow = prim::kPrimBlackmanWindow->name();
  static const auto &kHammingWindow = prim::kPrimHammingWindow->name();
  static const auto &kResizeArea = prim::kPrimResizeArea->name();
  static const auto &kExpand = prim::kPrimExpand->name();
  static const auto &kSspaddmm = prim::kPrimSspaddmm->name();
  static const auto &kBartlettWindow = prim::kPrimBartlettWindow->name();
  static const auto &kExtractGlimpse = prim::kPrimExtractGlimpse->name();
  static const auto &kTensorCopySlices = prim::kPrimTensorCopySlices->name();
  static const auto &kResizeNearestNeighborGrad = prim::kPrimResizeNearestNeighborGrad->name();
  static const auto &kSegmentMean = prim::kPrimSegmentMean->name();
  static const auto &kSegmentProd = prim::kPrimSegmentProd->name();
  static const auto &kStandardNormal = prim::kPrimStandardNormal->name();
  static const auto &kStandardLaplace = prim::kPrimStandardLaplace->name();
  static const auto &kCropAndResizeGradImage = prim::kPrimCropAndResizeGradImage->name();
  static const auto &kTraceGrad = prim::kPrimTraceGrad->name();
  static const auto &kSetSize = prim::kPrimSetSize->name();
  static const auto &kDynamicStitch = prim::kPrimDynamicStitch->name();
  static const auto &kSparseTensorDenseMatmul = prim::kPrimSparseTensorDenseMatmul->name();
  static const auto &kSparseMatrixTranspose = prim::kPrimSparseMatrixTranspose->name();
  static const auto &kSparseToDense = prim::kPrimSparseToDense->name();
  static const auto &kParameterizedTruncatedNormal = prim::kPrimParameterizedTruncatedNormal->name();
  // Common host depends.
  static PrimShapeDependMap host_depends{{prim::kPrimArgMax->name(), ShapeSet{1}},
                                         {prim::kPrimArgmin->name(), ShapeSet{1}},
                                         {kExtractGlimpse, ShapeSet{1}},
                                         {kMirrorPad, ShapeSet{1}},
                                         {kSegmentMax, ShapeSet{1}},
                                         {kSegmentMin, ShapeSet{1}},
                                         {kSegmentSum, ShapeSet{1}},
                                         {kSegmentMean, ShapeSet{1}},
                                         {kSegmentProd, ShapeSet{1}},
                                         {kUnsortedSegmentSum, ShapeSet{2}},
                                         {kFractionalAvgPoolGrad, ShapeSet{0}},
                                         {kUnsortedSegmentMin, ShapeSet{2}},
                                         {kUnsortedSegmentMax, ShapeSet{2}},
                                         {kUnsortedSegmentProd, ShapeSet{2}},
                                         {kMatrixDiagV3, ShapeSet{1, 2, 3, 4}},
                                         {kMatrixDiagPartV3, ShapeSet{1, 2}},
                                         {kMatrixSetDiagV3, ShapeSet{2}},
                                         {kGather, ShapeSet{2}},
                                         {kGatherV2, ShapeSet{2}},
                                         {kGatherD, ShapeSet{1}},
                                         {kSparseGatherV2, ShapeSet{2}},
                                         {kRangeV2, ShapeSet{0, 1, 2}},
                                         {kResizeBicubic, ShapeSet{1}},
                                         {kConv2DBackpropFilter, ShapeSet{2}},
                                         {kConv2DBackpropInput, ShapeSet{2}},
                                         {kCol2Im, ShapeSet{1}},
                                         {kOneHot, ShapeSet{1, 3}},
                                         {kDropoutGenMask, ShapeSet{0}},
                                         {kStridedSlice, ShapeSet{1, 2, 3}},
                                         {kStridedSliceGrad, ShapeSet{1, 2, 3, 4}},
                                         {kStridedSliceV2, ShapeSet{1, 2, 3}},
                                         {kStridedSliceV2Grad, ShapeSet{0}},
                                         {kTensorCopySlices, ShapeSet{2, 3, 4}},
                                         {kTile, ShapeSet{1}},
                                         {kTopK, ShapeSet{1}},
                                         {kReshape, ShapeSet{1}},
                                         {kResizeNearestNeighborV2, ShapeSet{1}},
                                         {kResizeNearestNeighborV2Grad, ShapeSet{1}},
                                         {kScatterNd, ShapeSet{2}},
                                         {kSparseToDenseV2, ShapeSet{1}},
                                         {prim::kPrimSparseTensorDenseMatmul->name(), ShapeSet{2}},
                                         {kSliceGrad, ShapeSet{2, 3}},
                                         {kFillV2, ShapeSet{0}},
                                         {kRandomCategorical, ShapeSet{1}},
                                         {kRandomGamma, ShapeSet{0, 1}},
                                         {kDynamicBroadcastTo, ShapeSet{1}},
                                         {kNonDeterministicInts, ShapeSet{0}},
                                         {prim::kPrimArgminV2->name(), ShapeSet{1}},
                                         {kAffineGrid, ShapeSet{1}},
                                         {prim::kPrimInplaceUpdateV2->name(), ShapeSet{1}},
                                         {kTruncatedNormal, ShapeSet{0}},
                                         {kRaggedRange, ShapeSet{0, 1, 2}},
                                         {kTransposeNOD, ShapeSet{1}},
                                         {kAdaptiveAvgPool3D, ShapeSet{1}},
                                         {kAdaptiveAvgPool3DGrad, ShapeSet{1}},
                                         {kResizeLinear1D, ShapeSet{1}},
                                         {kBlackmanWindow, ShapeSet{0}},
                                         {kHammingWindow, ShapeSet{0}},
                                         {kResizeArea, ShapeSet{1}},
                                         {kExpand, ShapeSet{1}},
                                         {kSspaddmm, ShapeSet{0, 2, 3, 5, 7}},
                                         {kBartlettWindow, ShapeSet{0}},
                                         {kResizeNearestNeighborGrad, ShapeSet{1}},
                                         {kTraceGrad, ShapeSet{1}},
                                         {kStandardNormal, ShapeSet{0}},
                                         {kStandardLaplace, ShapeSet{0}},
                                         {kCropAndResizeGradImage, ShapeSet{3}},
                                         {prim::kPrimCumSum->name(), ShapeSet{1}},
                                         {kAdaptiveMaxPool3DGrad, ShapeSet{1}},
                                         {kSetSize, ShapeSet{2}},
                                         {kDynamicStitch, ShapeSet{0}},
                                         {kSparseTensorDenseMatmul, ShapeSet{2}},
                                         {kSparseMatrixTranspose, ShapeSet{0}},
                                         {kParameterizedTruncatedNormal, ShapeSet{0}},
                                         {kSparseToDense, ShapeSet{2}}};
  return host_depends;
}

std::set<int64_t> GetValueDependArgIndices(const std::string &prim_name, size_t input_num) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  // Special dynamic shape depends for Ascend.
  if (device == kAscendDevice && prim_name == prim::kPrimTranspose->name()) {
    return {1};
  }

  std::set<int64_t> res = {};
  std::set<int64_t> ori = {};
  auto iter = GetHostDependsMap().find(prim_name);
  if (iter != GetHostDependsMap().end()) {
    ori = iter->second;
  }

  auto op_infer = GetPrimitiveInferImpl(std::make_shared<Primitive>(prim_name)).Get();
  if (op_infer != nullptr && ori.empty()) {
    ori = op_infer->GetValueDependArgIndices();
  }

  if (!ori.empty()) {
    (void)std::copy_if(ori.begin(), ori.end(), std::inserter(res, res.begin()),
                       [&](int64_t idx) { return idx < SizeToLong(input_num); });
  }

  return res;
}

std::set<int64_t> GetValueDependArgIndices(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->inputs().empty()) {
    MS_LOG(EXCEPTION) << "Invalid inputs";
  }
  auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->ToString();
  return GetValueDependArgIndices(prim_name, cnode->inputs().size() - 1);
}

void RegisterHostDependsImpl(const std::string &prim_name, const std::set<int64_t> &host_depends) {
  auto &host_depends_map = GetHostDependsMap();
  host_depends_map[prim_name] = host_depends;
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
    {prim::kPrimRealInner, R{InferImplReal, nullptr, true}},
    // Array
    {prim::kPrimArrayToScalar, R{InferImplArrayToScalar, nullptr, true}},
    {prim::kPrimBroadcastShape, R{InferImplBroadCastShape, nullptr, true}},
    {prim::kPrimUnique, R{InferImplUnique, nullptr, true}},
    {prim::kPrimUniqueWithPad, R{InferImplUniqueWithPad, nullptr, true}},
    {prim::kPrimUniqueGrad, R{InferImplUniqueGrad, nullptr, true}},
    {prim::kPrimEmbeddingLookup, R{InferImplEmbeddingLookup, nullptr, true}},
    {prim::kPrimSparseGatherV2, R{InferImplGatherV2, nullptr, true}},
    {prim::kPrimUnsortedSegmentMax, R{InferImplUnsortedSegmentMax, nullptr, true}},
    {prim::kPrimUnsortedSegmentMin, R{InferImplUnsortedSegmentMin, nullptr, true}},
    {prim::kPrimScatterAdd, R{InferImplScatterAdd, nullptr, true}},
    {prim::kPrimScatterSub, R{InferImplScatterSub, nullptr, true}},
    {prim::kPrimSubAndFilter, R{InferImplSubAndFilter, nullptr, true}},
    {prim::kPrimMapCacheIdx, R{InferImplMapCacheIdx, nullptr, true}},
    {prim::kPrimDynamicAssign, R{InferImplDynamicAssign, nullptr, true}},
    {prim::kPrimCacheSwapTable, R{InferImplCacheSwapTable, nullptr, true}},
    {prim::kPrimUpdateCache, R{InferImplUpdateCache, nullptr, true}},
    {prim::kPrimComputeAccidentalHits, R{InferImplComputeAccidentalHits, nullptr, true}},
    {prim::kPrimPadAndShift, R{InferImplPadAndShift, nullptr, true}},
    {prim::kPrimMapUniform, R{InferImplMapUniform, nullptr, true}},
    {prim::kPrimSplit, R{InferImplSplit, nullptr, true}},
    {prim::kPrimSequenceMask, R{InferImplSequenceMask, nullptr, true}},
    {prim::kPrimTensorCopySlices, R{InferImplTensorCopySlices, nullptr, true}},
    {prim::kPrimFlattenConcat, R{InferImplFlattenConcat, nullptr, true}},
    {prim::kPrimOCRRecognitionPreHandle, R{InferImplOCRRecognitionPreHandle, nullptr, true}},
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
    {prim::kPrimTupleLen, R{InferImplTupleLen, nullptr, true}},
    {prim::kPrimListLen, R{InferImplListLen, nullptr, true}},
    {prim::kPrimArrayLen, R{InferImplArrayLen, nullptr, true}},
    // NN
    {prim::kPrimPooling, R{InferImplPooling, nullptr, true}},
    {prim::kPrimPoolingGrad, R{InferImplPoolingGrad, nullptr, true}},
    {prim::kPrimBatchNorm, R{InferImplBatchNorm, nullptr, true}},
    {prim::kPrimBpropCut, R{InferImplBpropCut, nullptr, true}},
    {prim::kPrimBiasDropoutAdd, R{InferImplBiasDropoutAdd, nullptr, true}},
    {prim::kPrimSparseApplyProximalAdagrad, R{InferImplSparseApplyProximalAdagrad, nullptr, true}},
    // Others
    {prim::kPrimIdentity, R{InferImplIdentity, nullptr, true}},
    {prim::kPrimLoad, R{InferImplLoad, nullptr, true}},
    {prim::kPrimMutable, R{InferImplMutable, nullptr, true}},
    {prim::kPrimOpaquePredicate, R{ops::OpaquePredicateInfer, nullptr, true}},
    // Set impl to null as it will use PartialEvaluator;
    {prim::kPrimPartial, R{nullptr, nullptr, true}},
    {prim::kPrimEnvironCreate, R{InferImplEnvironCreate, nullptr, true}},
    {prim::kPrimEnvironGet, R{InferImplEnvironGet, nullptr, true}},
    {prim::kPrimEnvironSet, R{InferImplEnvironSet, nullptr, true}},
    {prim::kPrimEnvironAdd, R{InferImplEnvironAdd, nullptr, true}},
    {prim::kPrimEnvironDestroyAll, R{InferImplEnvironDestroyAll, nullptr, true}},
    {prim::kPrimStateSetItem, R{InferImplStateSetItem, nullptr, true}},
    {prim::kPrimDepend, R{InferImplDepend, nullptr, true}},
    {prim::kPrimUpdateState, R{InferImplUpdateState, nullptr, true}},
    // Debug
    {prim::kPrimDebug, R{InferImplDebug, nullptr, true}},
    // Dynamic shape testing
    {prim::kPrimGpuConvertToDynamicShape, R{InferImplGpuConvertToDynamicShape, nullptr, true}},
    // RowTensor
    {prim::kPrimMakeRowTensor, R{InferImplMakeRowTensor, nullptr, true}},
    {prim::kPrimRowTensorGetValues, R{InferImplRowTensorGetValues, nullptr, true}},
    {prim::kPrimRowTensorGetIndices, R{InferImplRowTensorGetIndices, nullptr, true}},
    {prim::kPrimRowTensorGetDenseShape, R{InferImplRowTensorGetDenseShape, nullptr, true}},
    {prim::kPrimRowTensorAdd, R{InferImplRowTensorAdd, nullptr, false}},
    // MapTensor
    {prim::kPrimMapTensorGetDefaultValue, R{InferImplMapTensorGetDefaultValue, nullptr, true}},
    // Comm Ops
    {prim::kPrimAllSwap, R{InferImplAllSwap, nullptr, true}},
    {prim::kPrimMemCpyAsync, R{InferImplMemCpyAsync, nullptr, true}},
    {prim::kPrimFusedPushWeight, R{nullptr, nullptr, true}},
    {prim::kPrimFusedPullWeight, R{nullptr, nullptr, true}},
    // RL Ops
    {prim::kPrimTensorArrayStack, R{InferImplTensorArrayStack, nullptr, true}},
    {prim::kPrimKMeansCentroids, R{InferImplKMeansCentroids, nullptr, true}},
  };
  return prim_eval_implement_map;
}

PrimitiveEvalImplMap &GetPrimitiveToBackendEvalImplMap() {
  using R = PrimitiveEvalImplMap::mapped_type;
  static PrimitiveEvalImplMap prim_backend_eval_implement_map = {
    {prim::kPrimMul, R{ops::MulInfer, nullptr, true}},
    {prim::kPrimMod, R{ops::ModInfer, nullptr, true}},
    {prim::kPrimAdd, R{ops::AddInfer, nullptr, false}},
    {prim::kPrimArgmin, R{ops::ArgMinInfer, nullptr, true}},
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
    {prim::kPrimBiasAddGrad, R{InferImplBiasAddGrad, nullptr, true}},
    {prim::kPrimReduceScatter, R{InferImplReduceScatter, nullptr, true}},
    {prim::kPrimCast, R{InferImplCast, nullptr, true}},
    {prim::kPrimExp, R{ops::ExpInfer, nullptr, true}},
    {prim::kPrimAllReduce, R{InferImplAllReduce, nullptr, true}},
    {prim::kPrimBroadcast, R{InferImplBroadcast, nullptr, true}},
    {prim::kPrimAllGather, R{InferImplAllGather, nullptr, true}},
    {prim::kPrimMinimum, R{InferImplMinimum, nullptr, true}},
    {prim::kPrimDivNoNan, R{InferImplDivNoNan, nullptr, true}},
    {prim::kPrimLinSpace, R{InferImplLinSpace, nullptr, true}},
    {prim::kPrimLess, R{InferImplLess, nullptr, true}},
    {prim::kPrimPad, R{InferImplPad, nullptr, true}},
    {prim::kPrimUnsortedSegmentSum, R{InferImplUnsortedSegmentSum, nullptr, true}},
    {prim::kPrimDiv, R{InferImplDiv, nullptr, true}},
    {prim::kPrimRealDiv, R{ops::RealDivInfer, nullptr, false}},
    {prim::kPrimTranspose, R{InferImplTranspose, nullptr, true}},
    {prim::kPrimTransposeNOD, R{InferImplTranspose, nullptr, true}},
    {prim::kPrimStridedSlice, R{ops::StridedSliceInfer, nullptr, true}},
    {prim::kPrimSlice, R{ops::SliceInfer, nullptr, true}},
    {prim::kPrimSliceGrad, R{ops::SliceGradInfer, nullptr, true}},
    {prim::kPrimConcat, R{InferImplConcat, nullptr, true}},
    {prim::kPrimConcatOffset, R{InferImplConcatOffset, nullptr, true}},
    {prim::kPrimTransData, R{InferImplTransData, nullptr, true}},
    {prim::kPrimTensorMove, R{InferImplTensorMove, nullptr, true}},
    {prim::kPrimLstm, R{ops::LstmInfer, nullptr, true}},
    {prim::kPrimStack, R{ops::StackInfer, nullptr, true}},
    {prim::kPrimRpcRecv, R{ops::RpcRecvInfer, nullptr, true}},
    {prim::kPrimRpcSend, R{ops::RpcSendInfer, nullptr, true}},
    {prim::kPrimAdamApplyOne, R{InferImplAdamApplyOne, nullptr, true}},
    {prim::kPrimAdamApplyOneWithDecay, R{InferImplAdamApplyOneWithDecay, nullptr, true}},
    {prim::kPrimTensorScatterUpdate, R{ops::TensorScatterArithmeticInfer, nullptr, true}},
    {prim::kPrimMaxPool, R{ops::MaxPoolInfer, nullptr, true}},
    {prim::kPrimMaxPoolGrad, R{ops::MaxPoolGradInfer, nullptr, true}}};
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

class OpInferCommon : public OpInferBase {
 public:
  OpInferCommon() = delete;
  OpInferCommon(const InferAbstractImpl &infer_impl, const InferValueImpl &infer_value_impl)
      : infer_impl_(infer_impl), infer_value_impl_(infer_value_impl) {}
  ~OpInferCommon() = default;

  BaseShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;
  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;
  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override;

 private:
  InferAbstractImpl infer_impl_{nullptr};
  InferValueImpl infer_value_impl_{nullptr};
};

BaseShapePtr OpInferCommon::InferShape(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  if (!infer_impl_) {
    return nullptr;
  }

  auto inferred_res = infer_impl_(nullptr, primitive, input_args);
  if (inferred_res == nullptr) {
    return nullptr;
  }

  return inferred_res->BuildShape();
}

TypePtr OpInferCommon::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  if (!infer_impl_) {
    return nullptr;
  }

  auto inferred_res = infer_impl_(nullptr, primitive, input_args);
  if (inferred_res == nullptr) {
    return nullptr;
  }

  return inferred_res->BuildType();
}

ValuePtr OpInferCommon::InferValue(const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) const {
  if (!infer_value_impl_) {
    return nullptr;
  }
  return infer_value_impl_(primitive, input_args);
}

AbstractBasePtr OpInferCommon::InferShapeAndType(const abstract::AnalysisEnginePtr &engine,
                                                 const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) const {
  if (!infer_impl_) {
    return nullptr;
  }

  return infer_impl_(engine, primitive, input_args);
}

StandardPrimitiveImplReg::StandardPrimitiveImplReg(const InferAbstractImpl &infer_abstract,
                                                   const InferValueImpl &infer_value, bool in_white_list) {
  op_infer_ = std::make_shared<OpInferCommon>(infer_abstract, infer_value);
  is_impl_infer_shape_and_type_ = infer_abstract != nullptr;
  is_impl_infer_value_ = infer_value != nullptr;
  in_white_list_ = in_white_list;
}

AbstractBasePtr StandardPrimitiveImplReg::InferShapeAndType(const abstract::AnalysisEnginePtr &engine,
                                                            const PrimitivePtr &primitive,
                                                            const std::vector<AbstractBasePtr> &input_args) const {
  if (op_infer_ == nullptr) {
    return nullptr;
  }

  return op_infer_->InferShapeAndType(engine, primitive, input_args);
}

BaseShapePtr StandardPrimitiveImplReg::InferShape(const PrimitivePtr &prim, const AbstractBasePtrList &args) const {
  if (op_infer_ == nullptr) {
    return nullptr;
  }

  return op_infer_->InferShape(prim, args);
}

TypePtr StandardPrimitiveImplReg::InferType(const PrimitivePtr &prim, const AbstractBasePtrList &args) const {
  if (op_infer_ == nullptr) {
    return nullptr;
  }

  return op_infer_->InferType(prim, args);
}

ValuePtr StandardPrimitiveImplReg::InferValue(const PrimitivePtr &prim, const AbstractBasePtrList &args) const {
  if (op_infer_ == nullptr) {
    return nullptr;
  }

  return op_infer_->InferValue(prim, args);
}
}  // namespace abstract
}  // namespace mindspore
