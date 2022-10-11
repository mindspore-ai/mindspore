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

#include "include/common/utils/utils.h"
#include <set>
#include <string>
namespace mindspore {
bool IsOneOfPosteriorOperator(const std::string &name) {
  const std::set<std::string> kPosteriorOperatorSet = {kPullOpName};

  auto iter = kPosteriorOperatorSet.find(name);
  return iter != kPosteriorOperatorSet.end();
}

bool IsOneOfCacheBlackList(const std::string &name) {
  const std::set<std::string> kOpCacheBlackList = {kUniformCandidateSamplerOpName, kInitDatasetQueueOpName,
                                                   kGetNextOpName};

  auto iter = kOpCacheBlackList.find(name);
  return iter != kOpCacheBlackList.end();
}

bool IsOneOfNotSupportMultiThreadExec(const std::string &name) {
  const std::set<std::string> kOpNotSupportMultiThreadExecList = {};
  auto iter = kOpNotSupportMultiThreadExecList.find(name);
  return iter != kOpNotSupportMultiThreadExecList.end();
}

bool IsOneOf3DFormat(const std::string &format) {
  const std::set<std::string> k3DFormatSet = {kOpFormat_NCDHW, kOpFormat_NDC1HWC0, kOpFormat_FRACTAL_Z_3D,
                                              kOpFormat_NDHWC, kOpFormat_DHWCN,    kOpFormat_DHWNC};

  auto iter = k3DFormatSet.find(format);
  return iter != k3DFormatSet.end();
}

bool IsOneOfNoPaddingFormat(const std::string &format) {
  const std::set<std::string> kNoPaddingFormatSet = {kOpFormat_ChannelLast, kOpFormat_FRAC_NZ, kOpFormat_FRACTAL_ZN_RNN,
                                                     kOpFormat_ND_RNN_BIAS};

  auto iter = kNoPaddingFormatSet.find(format);
  return iter != kNoPaddingFormatSet.end();
}

bool IsOneOfDynamicShapeConstInputToAttr(const std::string &name) {
  const std::set<std::string> DynamicShapeConstInputToAttr = {
    kCastOpName,      kExpandDimsOpName, kEmbeddingLookupOpName, kReduceMinOpName, kReduceMeanOpName,
    kReduceMaxOpName, kReduceAllOpName,  kReduceAnyOpName,       kConcatOpName,    kTransposeOpName};

  auto iter = DynamicShapeConstInputToAttr.find(name);
  return iter != DynamicShapeConstInputToAttr.end();
}

bool IsOneOfDynamicShapeConstInputToAttrCPU(const std::string &name) {
  const std::set<std::string> DynamicShapeConstInputToAttrCPU = {
    kCastOpName,      kExpandDimsOpName, kEmbeddingLookupOpName, kReduceMinOpName, kReduceMeanOpName, kReduceMaxOpName,
    kReduceAllOpName, kReduceAnyOpName,  kConcatOpName,          kReduceSumOpName, kTransposeOpName};

  auto iter = DynamicShapeConstInputToAttrCPU.find(name);
  return iter != DynamicShapeConstInputToAttrCPU.end();
}

bool IsOneOfDynamicShapeConstInputToAttrGPU(const std::string &name) {
  const std::set<std::string> DynamicShapeConstInputToAttrGPU = {
    kCastOpName,      kExpandDimsOpName, kReshapeOpName,    kEmbeddingLookupOpName, kTransposeOpName,
    kReduceSumOpName, kReduceMinOpName,  kReduceMeanOpName, kReduceMaxOpName,       kReduceAllOpName,
    kReduceAnyOpName, kConcatOpName,     kScatterNdOpName,  kGatherV2OpName,        kAvgPool3DGradOpName};

  auto iter = DynamicShapeConstInputToAttrGPU.find(name);
  return iter != DynamicShapeConstInputToAttrGPU.end();
}

bool IsOneOfCustomAkgType(const std::string &name) {
  const std::set<std::string> kCustomTypeAkg = {"ir_builder", "tvm_compute", "hybrid"};

  auto iter = kCustomTypeAkg.find(name);
  return iter != kCustomTypeAkg.end();
}

bool IsOneOfOperator(const std::string &name) {
  const std::set<std::string> kOptOperatorSet = {kMomentumOpName,
                                                 kApplyMomentumOpName,
                                                 kApplyAdadeltaOpName,
                                                 kApplyAdagradOpName,
                                                 kApplyAdagradDAName,
                                                 kApplyAdamOpName,
                                                 kApplyAdaMaxOpName,
                                                 kApplyAddSignOpName,
                                                 kApplyCenteredRMSPOpName,
                                                 kApplyFtrlOpName,
                                                 kApplyFtrlV2OpName,
                                                 kApplyGradientDescentOpName,
                                                 kApplyPowerSignOpName,
                                                 kApplyProximalAdagradOpName,
                                                 kApplyProximalGradientDescentOpName,
                                                 kApplyRMSPropOpName,
                                                 kAdamApplyOneWithDecayOpName,
                                                 kAdamApplyOneWithDecayAssignOpName,
                                                 kFusedAdamWeightDecayName,
                                                 kAdamWeightDecayName,
                                                 kFusedCastAdamWeightDecayName,
                                                 kFusedAdamName,
                                                 kFusedAdaFactorName,
                                                 kFusedAdaFactorWithGlobalNormName,
                                                 kFusedSparseAdamName,
                                                 kFusedMulApplyMomentumOpName,
                                                 kFusedWeightScaleApplyMomentum,
                                                 kFusedScaleApplyMomentum,
                                                 kApplyCenteredRMSPropOpName,
                                                 kFusedSparseFtrlName,
                                                 kFusedSparseProximalAdagradName,
                                                 kFusedSparseLazyAdamName,
                                                 kSparseApplyFtrlName,
                                                 kSparseApplyFtrlV2Name,
                                                 kSGDName,
                                                 kLARSUpdateName,
                                                 kCombineMomentumWeightOpName,
                                                 kCombineMomentumOpName,
                                                 kScatterAddOpName,
                                                 kScatterUpdateOpName,
                                                 kSparseApplyProximalAdagradOpName};

  auto iter = kOptOperatorSet.find(name);
  return iter != kOptOperatorSet.end();
}

bool IsOneOfComputeDepend(const std::string &name) {
  const std::set<std::string> kComputeDepend = {kUniqueOpName,
                                                kUniqueConsecutiveOpName,
                                                kComputeAccidentalHitsOpName,
                                                kSubAndFilterOpName,
                                                kPadAndShiftOpName,
                                                kCTCGreedyDecoderOpName,
                                                kMaskedSelectOpName,
                                                kDynamicStitchOpName,
                                                kGetNextOpName,
                                                kListDiffOpName,
                                                kNonMaxSuppressionV3OpName,
                                                kNonMaxSuppressionWithOverlapsOpName,
                                                kCoalesceOpName,
                                                kTruncatedNormal,
                                                kNonDeterministicInts,
                                                kFractionalAvgPoolGradOpName,
                                                kDenseToDenseSetOperation,
                                                kSegmentMaxOpName,
                                                kCSRSparseMatrixToSparseTensorOpName,
                                                kSegmentMinOpName,
                                                kLuUnpackOpName,
                                                kSegmentSumOpName,
                                                kResizeBicubicOpName,
                                                kResizeAreaOpName,
                                                kSegmentMeanOpName,
                                                kSegmentProdOpName,
                                                kNonZeroOpName,
                                                kSparseSparseMinimumOpName,
                                                kRpcRecvOpName,
                                                kAdaptiveMaxPool3DGradOpName};

  auto iter = kComputeDepend.find(name);
  return iter != kComputeDepend.end();
}

bool IsOneOfHWSpecialFormat(const std::string &format) {
  const std::set<std::string> kHWSpecialFormatSet = {
    kOpFormat_FRACTAL_Z_3D,   kOpFormat_NC1KHKWHWC0, kOpFormat_NC1HWC0,       kOpFormat_FRAC_NZ,
    kOpFormat_C1HWNCoC0,      kOpFormat_NC1HWC0_C04, kOpFormat_FRACTAL_Z_C04, kOpFormat_FRACTAL_ZN_LSTM,
    kOpFormat_FRACTAL_ZN_RNN, kOpFormat_NDC1HWC0,    kOpFormat_FRAC_Z};

  auto iter = kHWSpecialFormatSet.find(format);
  return iter != kHWSpecialFormatSet.end();
}

bool IsOneOfFormat(const std::string &format) {
  const std::set<std::string> kOpFormatList = {
    kOpFormat_DEFAULT,        kOpFormat_NC1KHKWHWC0,  kOpFormat_ND,
    kOpFormat_NCHW,           kOpFormat_NHWC,         kOpFormat_HWCN,
    kOpFormat_CHWN,           kOpFormat_NC1HWC0,      kOpFormat_FRAC_Z,
    kOpFormat_C1HWNCoC0,      kOpFormat_FRAC_NZ,      kOpFormat_NC1HWC0_C04,
    kOpFormat_FRACTAL_Z_C04,  kOpFormat_NDHWC,        kOpFormat_FRACTAL_ZN_LSTM,
    kOpFormat_FRACTAL_ZN_RNN, kOpFormat_ND_RNN_BIAS,  kOpFormat_NDC1HWC0,
    kOpFormat_NCDHW,          kOpFormat_FRACTAL_Z_3D, kOpFormat_DHWNC,
    kOpFormat_DHWCN};

  auto iter = kOpFormatList.find(format);
  return iter != kOpFormatList.end();
}

bool IsOneOfServerFormatC04(const std::string &format) {
  const std::set<std::string> kServerFormatC04List = {kOpFormat_NC1HWC0_C04, kOpFormat_FRACTAL_Z_C04};
  return kServerFormatC04List.find(format) != kServerFormatC04List.end();
}
}  // namespace mindspore
