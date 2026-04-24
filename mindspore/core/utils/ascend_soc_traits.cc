/**
 * Copyright 2026 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "utils/ascend_soc_traits.h"

#include <algorithm>
#include <unordered_map>
#include <vector>

#include "utils/ms_context.h"

namespace mindspore {
namespace ascend_soc {
namespace {
using FeatureList = std::vector<std::string>;

FeatureList SupportedFeaturesForAscend910() {
  return {
    kAscendFeatureJitLevelO2DefaultAscendJit,
    kAscendFeatureRepeatInterleaveOpTrainClassicConstraint,
    kAscendFeatureGraphKernelStitchFusionDefaultTrainClassic,
    kAscendFeatureParallelDeviceListCheckTrainClassic,
    kAscendFeatureRuntimeDisableGeKernelTrainClassic,
    kAscendFeatureGeFormatModeGraphSinkDefaultTrainClassic,
    kAscendFeatureDeviceContextGeInitBeforeDistributedTrainClassic,
    kAscendFeatureJitConfigBackendGeDefaultEmptyJit,
    kAscendFeaturePyQueryTrainClassicAcceleratorLine,
    kAscendFeatureAmpLossScaleTrainClassicTarget,
    kAscendFeatureAoeAscend9ServerLine,
    kAscendFeatureMc2MatmulReduceScatterAclnnSupported,
    kAscendFeatureMc2AllgatherMatmulAclnnSupported,
    kAscendFeatureAtbInplaceGroupedMatmulRunOperatorLaunch,
  };
}

FeatureList SupportedFeaturesForAscend910b() {
  return {
    kAscendFeatureAclOverflowModeDeviceContextInitSuccessor,
    kAscendFeatureAclOverflowModeGeBackendInitSuccessor,
    kAscendFeatureInternalKernelNzFormatMatmulSuccessor,
    kAscendFeatureFusionMatmulElemwiseNonFastgeluSuccessor,
    kAscendFeaturePyQueryTrainSuccessorFamilyLine,
    kAscendFeatureAvgPool2dPadConstraintTrainSuccessor,
    kAscendFeatureTrainHighPrecisionDtypeLine,
    kAscendFeatureGeSessionPrecisionMustKeepOriginDtypeCloud,
    kAscendFeatureAoeOnlinePrecisionMustKeepOriginDtype,
    kAscendFeatureParallelOverlapGradCommMainCloudTrain,
    kAscendFeatureParallelOverlapRecomputeAllgatherFaGradMainCloudTrain,
    kAscendFeatureGraphKernelDvm,
    kAscendFeatureFusionMatmulElemwiseMatmulElemwisePass,
    kAscendFeatureFusionMatmulElemwiseTransposeExtPass,
    kAscendFeatureInferBoost910bFamilyFusionMatmulSigmoidAdd,
    kAscendFeatureInferBoost910bFamilyFusionMatmulSigmoidCastAdd,
    kAscendFeatureInferBoost910bFamilyFusionTransposeBatchMatmulTranspose,
    kAscendFeatureInferBoost910bFamilyFusionSwigluDynamicQuant,
    kAscendFeatureInferBoost910bFamilyFusionSwigluReshapeDynamicQuant,
    kAscendFeatureInferBoost910bFamilyFusionMoeInitRoutingDynQuantV2,
    kAscendFeatureInternalKernelAclGraph,
    kAscendFeatureMc2Lccl,
    kAscendFeatureMemoryDefault64gLine,
    kAscendFeatureDeviceCapacity64gLine,
    kAscendFeatureKbkTransData910bNonCustom,
    kAscendFeatureGeExecFormatModePreferAltSocLine,
    kAscendFeatureShapeReshapeFusionNonGraphRunAltSocLine,
    kAscendFeatureDvmCommCollective910bNormalizedPath,
    kAscendFeatureMatmulAllreduceAddRmsnormFusion910bNormalizedLine,
    kAscendFeatureMc2MatmulReduceScatterAclnnSupported,
    kAscendFeatureMc2AllgatherMatmulAclnnSupported,
    kAscendFeatureAtbInplaceGroupedMatmulRunOperatorLaunch,
    kAscendFeatureAoeAscend9ServerLine,
    kAscendFeatureMindDataDvppSupportedSocLine,
    kAscendFeatureMemoryVmmDefaultEnabledSocLine,
  };
}

FeatureList SupportedFeaturesForAscend910_93() {
  return {
    kAscendFeatureAclOverflowModeDeviceContextInitSuccessor,
    kAscendFeatureAclOverflowModeGeBackendInitSuccessor,
    kAscendFeatureInternalKernelNzFormatMatmulSuccessor,
    kAscendFeatureFusionMatmulElemwiseNonFastgeluSuccessor,
    kAscendFeaturePyQueryTrainSuccessorFamilyLine,
    kAscendFeatureAvgPool2dPadConstraintTrainSuccessor,
    kAscendFeatureTrainHighPrecisionDtypeLine,
    kAscendFeatureGeSessionPrecisionMustKeepOriginDtypeCloud,
    kAscendFeatureAoeOnlinePrecisionMustKeepOriginDtype,
    kAscendFeatureParallelOverlapGradCommMainCloudTrain,
    kAscendFeatureParallelOverlapRecomputeAllgatherFaGradMainCloudTrain,
    kAscendFeatureMc2Lccl91093Variant,
    kAscendFeatureGraphKernelDvm,
    kAscendFeatureFusionMatmulElemwiseMatmulElemwisePass,
    kAscendFeatureFusionMatmulElemwiseTransposeExtPass,
    kAscendFeatureInferBoost910bFamilyFusionMatmulSigmoidAdd,
    kAscendFeatureInferBoost910bFamilyFusionMatmulSigmoidCastAdd,
    kAscendFeatureInferBoost910bFamilyFusionTransposeBatchMatmulTranspose,
    kAscendFeatureInferBoost910bFamilyFusionSwigluDynamicQuant,
    kAscendFeatureInferBoost910bFamilyFusionSwigluReshapeDynamicQuant,
    kAscendFeatureInferBoost910bFamilyFusionMoeInitRoutingDynQuantV2,
    kAscendFeatureInternalKernelAclGraph,
    kAscendFeatureMc2Lccl,
    kAscendFeatureMemoryDefault64gLine,
    kAscendFeatureDeviceCapacity64gLine,
    kAscendFeatureKbkTransData910bNonCustom,
    kAscendFeatureGeExecFormatModePreferAltSocLine,
    kAscendFeatureShapeReshapeFusionNonGraphRunAltSocLine,
    kAscendFeatureDvmCommCollective910bNormalizedPath,
    kAscendFeatureMatmulAllreduceAddRmsnormFusion910bNormalizedLine,
    kAscendFeatureAtbInplaceGroupedMatmulRunOperatorLaunch,
    kAscendFeatureAoeAscend9ServerLine,
    kAscendFeatureMindDataDvppSupportedSocLine,
    kAscendFeatureMemoryVmmDefaultEnabledSocLine,
  };
}

FeatureList SupportedFeaturesForAscend310p() {
  return {
    kAscendFeatureFusionMatmulElemwiseMatmulElemwisePass,
    kAscendFeatureFusionMatmulElemwiseTransposeExtPass,
    kAscendFeatureInferBoostQbmmElemwise,
    kAscendFeatureAscendResManagerInfer310SkipsHcclTimeoutApis,
    kAscendFeaturePyQueryInfer310FamilyLine,
    kAscendFeatureKbkTransData310pInternalKernel,
    kAscendFeatureGeExecFormatModePreferAltSocLine,
    kAscendFeatureShapeReshapeFusionNonGraphRunAltSocLine,
    kAscendFeatureMemoryTotalInfer310pLine,
    kAscendFeatureInferCommSkipInterNodeHcclGroup,
    kAscendFeatureCpuCollectiveAllreduceLauncherInfer310p,
    kAscendFeatureInternalKernelMatmulNzFormatInfer310p,
    kAscendFeatureInferBoostFormatIdentityTransDataInternal,
    kAscendFeatureInferBoostGroupedMatmulInternalKernel310p,
    kAscendFeatureInternalOpListDefaultInfer310p,
    kAscendFeatureMc2MatmulReduceScatterAclnnSupported,
    kAscendFeatureAtbInplaceGroupedMatmulRunOperatorLaunch,
  };
}

FeatureList SupportedFeaturesForAscend310b() {
  return {
    kAscendFeatureAscendResManagerInfer310SkipsHcclTimeoutApis,
    kAscendFeaturePyQueryInfer310FamilyLine,
    kAscendFeatureGeExecFormatModePreferAltSocLine,
    kAscendFeatureShapeReshapeFusionNonGraphRunAltSocLine,
    kAscendFeatureMc2MatmulReduceScatterAclnnSupported,
    kAscendFeatureMc2AllgatherMatmulAclnnSupported,
    kAscendFeatureAtbInplaceGroupedMatmulRunOperatorLaunch,
  };
}

FeatureList SupportedFeaturesForAscend950() {
  return {
    kAscendFeatureTrainHighPrecisionDtypeLine,
    kAscendFeatureGeSessionPrecisionMustKeepOriginDtypeCloud,
    kAscendFeatureAoeOnlinePrecisionMustKeepOriginDtype,
    kAscendFeatureParallelOverlapGradCommMainCloudTrain,
    kAscendFeatureParallelOverlapRecomputeAllgatherFaGradMainCloudTrain,
    kAscendFeatureGeExecFormatModePreferAltSocLine,
    kAscendFeatureShapeReshapeFusionNonGraphRunAltSocLine,
    kAscendFeatureMemoryTotalDevice950SkuTable,
    kAscendFeatureAtbInplaceGroupedMatmulSkipWorkspaceInit,
    kAscendFeatureAoeAscend9ServerLine,
    kAscendFeatureMindDataDvppSupportedSocLine,
    kAscendFeatureMemoryVmmDefaultEnabledSocLine,
    kAscendFeatureMc2MatmulReduceScatterAclnnSupported,
    kAscendFeatureMc2AllgatherMatmulAclnnSupported,
  };
}

const std::unordered_map<std::string, FeatureList> &SocToSupportedFeatures() {
  static const std::unordered_map<std::string, FeatureList> registry = {
    {kAscendVersion910, SupportedFeaturesForAscend910()},       {kAscendVersion910b, SupportedFeaturesForAscend910b()},
    {kAscendVersion910_93, SupportedFeaturesForAscend910_93()}, {kAscendVersion310p, SupportedFeaturesForAscend310p()},
    {kAscendVersion310b, SupportedFeaturesForAscend310b()},     {kAscendVersion950, SupportedFeaturesForAscend950()},
  };
  return registry;
}
}  // namespace

bool SocSupportsAscendFeatureForSocVersion(const std::string &soc_version, const std::string &feature_name) {
  if (soc_version.empty()) {
    return false;
  }
  const auto &registry = SocToSupportedFeatures();
  auto iter = registry.find(soc_version);
  if (iter == registry.end()) {
    return false;
  }
  return std::any_of(iter->second.begin(), iter->second.end(),
                     [&feature_name](const std::string &feat) { return feat == feature_name; });
}

bool SocSupportsAscendFeature(const std::string &feature_name) {
  auto ctx = MsContext::GetInstance();
  if (ctx == nullptr) {
    return false;
  }
  return SocSupportsAscendFeatureForSocVersion(ctx->ascend_soc_version(), feature_name);
}

}  // namespace ascend_soc
}  // namespace mindspore
