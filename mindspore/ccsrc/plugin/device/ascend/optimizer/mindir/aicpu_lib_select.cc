/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/optimizer/mindir/aicpu_lib_select.h"
#include "plugin/device/ascend/kernel/aicpu/aicpu_util.h"
#include <set>
#include <string>
#include "include/common/utils/utils.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
const AnfNodePtr AICpuLibSelectPass::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                             const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(equiv);

  static const std::set<std::string> kAICpuOpNames = {kDropoutGenMaskOpName,
                                                      kEnvironCreateOpName,
                                                      kEnvironSetOpName,
                                                      kEnvironGetOpName,
                                                      kEnvironDestroyAllOpName,
                                                      kPriorityReplayBufferCreate,
                                                      kPriorityReplayBufferPush,
                                                      kPriorityReplayBufferSample,
                                                      kPriorityReplayBufferUpdate,
                                                      kPriorityReplayBufferDestroy,
                                                      kReservoirReplayBufferCreate,
                                                      kReservoirReplayBufferPush,
                                                      kReservoirReplayBufferSample,
                                                      kReservoirReplayBufferDestroy,
                                                      kGatherDGradV2OpName,
                                                      kConcatOffsetOpName,
                                                      kSliceGradOpName,
                                                      kRandomShuffleOpName,
                                                      kRangeOpName,
                                                      kQuantDTypeCastOpName,
                                                      kFSEDecodeOpName};
  static const std::set<std::string> kMigrateAicpuKernelOps = {mindspore::kAdaptiveAvgPool2dOpName,
                                                               mindspore::kAdaptiveAvgPool2dGradOpName,
                                                               mindspore::kBucketizeOpName,
                                                               mindspore::kCacheSwapTableOpName,
                                                               mindspore::kCauchyOpName,
                                                               mindspore::kChannelShuffleOpName,
                                                               mindspore::kCholeskyGradOpName,
                                                               mindspore::kCholeskyInverseOpName,
                                                               mindspore::kCholeskySolveOpName,
                                                               mindspore::kCol2imOpName,
                                                               mindspore::kCombinedNonMaxSuppressionOpName,
                                                               mindspore::kComplexOpName,
                                                               mindspore::kComplexAbsOpName,
                                                               mindspore::kConcatOpName,
                                                               mindspore::kCosOpName,
                                                               mindspore::kCountNonZeroOpName,
                                                               mindspore::kCumulativeLogsumexpOpName,
                                                               mindspore::kCumprodOpName,
                                                               mindspore::kCSRSparseMatrixToDenseOpName,
                                                               mindspore::kCSRSparseMatrixToSparseTensorOpName,
                                                               mindspore::kDataFormatVecPermuteOpName,
                                                               mindspore::kFillOpName,
                                                               mindspore::kLogMatrixDeterminantOpName,
                                                               mindspore::kMatrixSolveLsOpName,
                                                               mindspore::kMedianOpName,
                                                               mindspore::kACosGradOpName,
                                                               mindspore::kAcoshGradOpName,
                                                               mindspore::kAdaptiveAvgPool3DOpName,
                                                               mindspore::kAdaptiveAvgPool3DGradOpName,
                                                               mindspore::kAdaptiveMaxPool2DGradOpName,
                                                               mindspore::kAdaptiveMaxPool3DOpName,
                                                               mindspore::kAdaptiveMaxPool3DGradOpName,
                                                               mindspore::kAddNOpName,
                                                               mindspore::kAddV2OpName,
                                                               mindspore::kAdjustContrastv2OpName,
                                                               mindspore::kAdjustHueOpName,
                                                               mindspore::kAdjustSaturationOpName,
                                                               mindspore::kAffineGridGradOpName,
                                                               mindspore::kAngleOpName,
                                                               mindspore::kArgmaxOpName,
                                                               mindspore::kArgMaxWithValueOpName,
                                                               mindspore::kArgMinOpName,
                                                               mindspore::kArgMinWithValueOpName,
                                                               mindspore::KAsinGradOpName,
                                                               mindspore::KAsinhGradOpName,
                                                               mindspore::kAvgPoolOpName,
                                                               mindspore::kAvgPoolGradOpName,
                                                               mindspore::kBartlettWindowOpName,
                                                               mindspore::kBatchNormGradGradOpName,
                                                               mindspore::kBiasAddOpName,
                                                               mindspore::kBiasAddGradOpName,
                                                               mindspore::kBincountOpName,
                                                               mindspore::kBlackmanWindowOpName,
                                                               mindspore::kBroadcastOpName,
                                                               mindspore::kMedianGradOpName,
                                                               mindspore::kNMSWithMaskOpName,
                                                               mindspore::kReduceSumOpName,
                                                               mindspore::kSpaceToDepthOpName,
                                                               mindspore::kSparseAddmmOpName,
                                                               mindspore::kSparseApplyAdagradDAOpName,
                                                               mindspore::kSparseApplyCenteredRMSPropOpName,
                                                               mindspore::kSparseApplyMomentumOpName,
                                                               mindspore::kSparseApplyProximalGradientDescentOpName,
                                                               mindspore::kSparseConcatOpName,
                                                               mindspore::kSparseDenseCwiseAddOpName,
                                                               mindspore::kSparseDenseCwiseDivOpName,
                                                               mindspore::kSparseDenseCwiseMulOpName,
                                                               mindspore::kSparseMatrixMatMulOpName,
                                                               mindspore::kSparseMatrixNNZOpName,
                                                               mindspore::kSparseMatrixTransposeOpName,
                                                               mindspore::kSparseFillEmptyRowsGradOpName,
                                                               mindspore::kSparseReshapeOpName,
                                                               mindspore::kSparseSegmentSqrtNGradOpName,
                                                               mindspore::kSparseSegmentSqrtNWithNumSegmentsOpName,
                                                               mindspore::kSparseSoftmaxCrossEntropyWithLogitsOpName,
                                                               mindspore::kSparseSparseMaximumOpName,
                                                               mindspore::kSparseSparseMinimumOpName,
                                                               mindspore::kSparseSegmentSumWithNumSegmentsOpName,
                                                               mindspore::kSplitOpName,
                                                               mindspore::kSqrtOpName,
                                                               mindspore::kSqrtGradOpName,
                                                               mindspore::kTanhOpName,
                                                               mindspore::kTileOpName,
                                                               mindspore::kTridiagonalMatMulOpName,
                                                               mindspore::kTripletMarginLossOpName,
                                                               mindspore::kTransposeOpName,
                                                               mindspore::kTriuIndicesOpName,
                                                               mindspore::kTrilIndicesOpName,
                                                               mindspore::kUnpackOpName,
                                                               mindspore::kUnravelIndexOpName,
                                                               mindspore::kUnsortedSegmentSumOpName,
                                                               mindspore::kUpperBoundOpName,
                                                               mindspore::kXlogyOpName,
                                                               mindspore::kXdivyOpName,
                                                               mindspore::kFFTWithSizeOpName,
                                                               mindspore::kHistogramDOpName,
                                                               mindspore::kIm2colOpName,
                                                               mindspore::kGatherNdOpName,
                                                               mindspore::kScatterNdOpName,
                                                               mindspore::kScatterNdUpdateOpName,
                                                               mindspore::kTensorScatterUpdateOpName,
                                                               mindspore::kIsNanOpName,
                                                               mindspore::kMatrixDeterminantOpName,
                                                               mindspore::kMatrixLogarithmOpName,
                                                               mindspore::kMatrixSetDiagV3OpName,
                                                               mindspore::kMultinomialOpName,
                                                               mindspore::kNanToNumOpName,
                                                               mindspore::kQrOpName,
                                                               mindspore::kResizeBicubicOpName,
                                                               mindspore::kNuclearNormOpName,
                                                               mindspore::kQuantileOpName,
                                                               mindspore::kSparseSegmentSqrtNOpName,
                                                               mindspore::kUnsortedSegmentProdOpName,
                                                               mindspore::kExpOpName,
                                                               mindspore::kMatrixTriangularSolveOpName,
                                                               mindspore::kMaximumGradGradOpName,
                                                               mindspore::kMaxPoolOpName,
                                                               mindspore::kMinimumGradGradOpName,
                                                               mindspore::kMulNoNanOpName,
                                                               mindspore::kMultilabelMarginLossGradOpName,
                                                               mindspore::kNthElementOpName,
                                                               mindspore::kNonMaxSuppressionWithOverlapsOpName,
                                                               mindspore::kOneHotOpName,
                                                               mindspore::kOrgqrOpName,
                                                               mindspore::kPackOpName,
                                                               mindspore::kParameterizedTruncatedNormalOpName,
                                                               mindspore::kPolarOpName,
                                                               mindspore::kPdistGradOpName,
                                                               mindspore::kRaggedRangeOpName,
                                                               mindspore::kRaggedTensorToSparseOpName,
                                                               mindspore::kRaggedTensorToTensorOpName,
                                                               mindspore::kReciprocalOpName,
                                                               mindspore::kReciprocalGradOpName,
                                                               mindspore::kReduceMeanOpName,
                                                               mindspore::kReduceProdOpName,
                                                               mindspore::kReluOpName,
                                                               mindspore::kReverseV2OpName,
                                                               mindspore::kRGBToHSVOpName,
                                                               mindspore::kRsqrtGradOpName,
                                                               mindspore::kSampleDistortedBoundingBoxExt2OpName,
                                                               mindspore::kScaleAndTranslateGradOpName,
                                                               mindspore::kScatterNdOpName,
                                                               mindspore::kScatterNdUpdateOpName,
                                                               mindspore::kSelectOpName,
                                                               mindspore::kSelfAdjointEigOpName,
                                                               mindspore::kSinOpName,
                                                               mindspore::kSincOpName,
                                                               mindspore::kSinhOpName,
                                                               mindspore::kSmoothL1LossGradV2OpName,
                                                               mindspore::kSmoothL1LossV2OpName,
                                                               mindspore::kSignOpName,
                                                               mindspore::kCheckNumericsOpName,
                                                               mindspore::kFloorDivOpName,
                                                               mindspore::kLog1pOpName,
                                                               mindspore::kMulOpName,
                                                               mindspore::kConjOpName,
                                                               mindspore::kZerosLikeOpName,
                                                               mindspore::kMatrixBandPartOpName,
                                                               mindspore::kDenseToCSRSparseMatrixOpName,
                                                               mindspore::kDenseToSparseSetOperation,
                                                               mindspore::kDiagOpName,
                                                               mindspore::kDiagonalOpName,
                                                               mindspore::kDiagPartOpName,
                                                               mindspore::kEigOpName,
                                                               mindspore::kEyeOpName,
                                                               mindspore::kMaximumOpName,
                                                               mindspore::kMinimumOpName,
                                                               mindspore::kFractionalAvgPoolOpName,
                                                               mindspore::kFractionalAvgPoolGradOpName,
                                                               mindspore::kFractionalMaxPoolOpName,
                                                               mindspore::kFractionalMaxPoolGradOpName,
                                                               mindspore::kFractionalMaxPoolGradWithFixedKsizeOpName,
                                                               mindspore::kGatherNdOpName,
                                                               mindspore::kGcdOpName,
                                                               mindspore::kGeqrfOpName,
                                                               mindspore::kHardSigmoidOpName,
                                                               mindspore::kHardSigmoidGradOpName,
                                                               mindspore::kHeavisideOpName,
                                                               mindspore::kHypotOpName,
                                                               mindspore::kIdentityNOpName,
                                                               mindspore::kIndexFillOpName,
                                                               mindspore::kKLDivOpName,
                                                               mindspore::kKlDivLossGradOpName,
                                                               mindspore::kLcmOpName,
                                                               mindspore::kLogitOpName,
                                                               mindspore::kLogitGradOpName,
                                                               mindspore::kLowerBoundOpName,
                                                               mindspore::kLstsqOpName,
                                                               mindspore::kLuUnpackOpName,
                                                               mindspore::kLuUnpackGradOpName,
                                                               mindspore::kMatMulOpName,
                                                               mindspore::kMatrixExpOpName,
                                                               mindspore::kPadV3GradOpName,
                                                               mindspore::kPadV3OpName,
                                                               mindspore::kLogicalXorOpName,
                                                               mindspore::kLogNormalReverseOpName};

  static const std::string kEnvOpSoNames = "mindspore_aicpu_kernels";
  static const std::string kCpuKernelSoName = "mindspore_cpu_kernels";

  if (!node->isa<CNode>()) {
    return node;
  }
  auto kernel_name = common::AnfAlgo::GetCNodeName(node);
  if (kernel::kOpNameToAicpuOpNameMap.find(kernel_name) != kernel::kOpNameToAicpuOpNameMap.end()) {
    kernel_name = kernel::kOpNameToAicpuOpNameMap.at(kernel_name);
  }
  if (kAICpuOpNames.find(kernel_name) != kAICpuOpNames.end()) {
    common::AnfAlgo::SetNodeAttr(kAttrCustAicpu, MakeValue(kEnvOpSoNames), node);
  }
  if (kMigrateAicpuKernelOps.find(kernel_name) != kMigrateAicpuKernelOps.end()) {
    common::AnfAlgo::SetNodeAttr(kAttrCustAicpu, MakeValue(kCpuKernelSoName), node);
  }

  return node;
}
}  // namespace opt
}  // namespace mindspore
