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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_MINDIR_REG_ASCEND_VM_OP_ADAPTATION_INFO_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_MINDIR_REG_ASCEND_VM_OP_ADAPTATION_INFO_H_

#include "backend/common/optimizer/op_adaptation_info_factory.h"
#include "plugin/device/ascend/optimizer/mindir/reg_ascend_vm_op_adaptation_funcs.h"
#include "include/common/utils/utils.h"

namespace mindspore::opt {
#define REG_ASCEND_VM_OP_ADAPTATION_INFO(me_op_name) REG_OP_ADAPTATION_INFO(me_op_name, kAscendDevice, true)

REG_ASCEND_VM_OP_ADAPTATION_INFO(kCOO2CSROpName).set_input_attr_info(1);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kCSR2COOOpName).set_input_attr_info(1);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kCSRDivOpName).set_input_attr_info(3);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kCSRGatherOpName).set_input_attr_info(3);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kCSRMMOpName).set_input_attr_info(3);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kCSRMulOpName).set_input_attr_info(3);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kCSRMVOpName).set_input_attr_info(3);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kCSRReduceSumOpName).set_input_attr_info(3).set_input_attr_info(4);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kErfOpName).set_input_attr_info(1);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kEyeOpName).set_input_attr_info(0).set_input_attr_info(1).set_input_attr_info(2);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kFlattenGradOpName).set_input_attr_info(1);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kMeanGradOpName).set_input_attr_info(1);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kPullWeightOpName).set_input_attr_info(1).set_input_attr_info(2);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kPushOpName).set_input_attr_info(1);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kPushWeightOpName).set_input_attr_info(1).set_input_attr_info(2);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kROIAlignGradName).set_input_attr_info(2);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kSimpleMeanGradOpName).set_input_attr_info(1);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kSubscalarOpName).set_input_attr_info(1);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kGatherDGradV2OpName).set_input_attr_info(1);

REG_ASCEND_VM_OP_ADAPTATION_INFO(kReshapeOpName).set_input_attr_info(1, "listInt");
REG_ASCEND_VM_OP_ADAPTATION_INFO(kTensorCopySlicesOpName)
  .set_input_attr_info(2, "listInt")
  .set_input_attr_info(3, "listInt")
  .set_input_attr_info(4, "listInt");
REG_ASCEND_VM_OP_ADAPTATION_INFO(kAdaptiveMaxPool2DOpName).set_backend_op_name(kAdaptiveMaxPool2dOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kApplyAdadeltaOpName).set_backend_op_name(kApplyAdadeltaDOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kApplyAdagradOpName).set_backend_op_name(kApplyAdagradDOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kApplyAdaMaxOpName).set_backend_op_name(kApplyAdaMaxDOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kApplyAddSignOpName).set_backend_op_name(kApplyAddSignDOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kApplyCenteredRMSPropOpName).set_backend_op_name(kApplyCenteredRMSPropDOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kApplyFtrlOpName).set_backend_op_name(kApplyFtrlDOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kApplyFtrlV2OpName).set_backend_op_name(kApplyFtrlV2DOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kApplyKerasMomentumOpName).set_backend_op_name(kApplyKerasMomentumDOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kApplyMomentumOpName).set_backend_op_name(kApplyMomentumDOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kApplyPowerSignOpName).set_backend_op_name(kApplyPowerSignDOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kApplyProximalAdagradOpName).set_backend_op_name(kApplyProximalAdagradDOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kAvgPool3DOpName).set_backend_op_name(kAvgPool3DDOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kACosGradOpName).set_backend_op_name(kAcosGradOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kBCEWithLogitsLossOpName).set_backend_op_name(kSigmoidCrossEntropyWithLogitsV2OpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kBNInferenceOpName).set_backend_op_name(kBNInferenceDOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kCeLUOpName).set_backend_op_name(kCeluV2OpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kDepthwiseConv2dNativeOpName).set_backend_op_name(kDepthwiseConv2DOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kDiagPartOpName).set_backend_op_name(kDiagPartDOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kDiagOpName).set_backend_op_name(kDiagDOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kDivOpName).set_backend_op_name(kTruncateDivOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kDropoutDoMaskOpName).set_backend_op_name(kDropOutDoMaskOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kFastGeLUOpName).set_backend_op_name(kFastGeluOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kFastGeLUGradOpName).set_backend_op_name(kFastGeluGradOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kGeLUOpName).set_backend_op_name(kGeluOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kGeLUGradOpName).set_backend_op_name(kGeluGradOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kHShrinkOpName).set_backend_op_name(kHardShrinkOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kHShrinkGradOpName).set_backend_op_name(kHardShrinkGradOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kHSigmoidOpName).set_backend_op_name(kHardSigmoidOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kHSigmoidGradOpName).set_backend_op_name(kHardSigmoidGradOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kHSwishOpName).set_backend_op_name(kHardSwishOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kHSwishGradOpName).set_backend_op_name(kHardSwishGradOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kIndexAddOpName).set_backend_op_name(kInplaceIndexAddOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kInplaceAddOpName).set_backend_op_name(kInplaceAddDOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kInplaceSubOpName).set_backend_op_name(kInplaceSubDOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kInplaceUpdateOpName).set_backend_op_name(kInplaceUpdateDOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kIOUOpName).set_backend_op_name(kIouOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kKLDivLossOpName).set_backend_op_name(kKLDivOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kLARSUpdateOpName).set_backend_op_name(kLarsV2UpdateOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kLinSpaceOpName).set_backend_op_name(kLinSpaceDOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kLogSoftmaxOpName).set_backend_op_name(kLogSoftmaxV2OpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kMatrixDiagOpName).set_backend_op_name(kMatrixDiagDOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kMatrixDiagPartOpName).set_backend_op_name(kMatrixDiagPartDOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kMatrixSetDiagOpName).set_backend_op_name(kMatrixSetDiagDOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kMaxPool3DGradGradOpName).set_backend_op_name(kMaxPool3DGradGradDOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kIm2ColOpName).set_backend_op_name(kIm2colOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kNewIm2ColOpName).set_backend_op_name(kIm2colOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kParallelResizeBilinearOpName).set_backend_op_name(kSyncResizeBilinearV2OpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kParallelResizeBilinearGradOpName)
  .set_target_op_name(kSyncResizeBilinearV2GradOpName)
  .set_input_attr_info(2, "listInt");
REG_ASCEND_VM_OP_ADAPTATION_INFO(kPReLUOpName).set_backend_op_name(kPReluOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kPSROIPoolingOpName).set_backend_op_name(kPSROIPoolingV2OpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kPSROIPoolingGradOpName).set_backend_op_name(kPSROIPoolingGradV2DOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kPReLUGradOpName).set_backend_op_name(kPReluGradOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kReLUOpName).set_backend_op_name(kReluOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kReLU6OpName).set_backend_op_name(kRelu6OpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kReLU6GradOpName).set_backend_op_name(kRelu6GradOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kReLUV2OpName).set_backend_op_name(kReluV2OpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kResizeBilinearOpName)
  .set_target_op_name(kResizeBilinearV2DOpName)
  .set_need_tbe_check_supported(true);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kResizeBilinearGradOpName).set_backend_op_name(kResizeBilinearV2GradOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kSeLUOpName).set_backend_op_name(kSeluOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kSmoothL1LossOpName).set_backend_op_name(kSmoothL1LossV2OpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kSmoothL1LossGradOpName).set_backend_op_name(kSmoothL1LossGradV2OpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kSoftmaxOpName).set_backend_op_name(kSoftmaxV2OpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kSoftmaxV2WithDropoutDoMaskV3OpName)
  .set_backend_op_name(kSoftmaxV2WithDropOutDoMaskV3DOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kSparseApplyAdagradOpName).set_backend_op_name(kSparseApplyAdagradDOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kSparseApplyProximalAdagradOpName)
  .set_backend_op_name(kSparseApplyProximalAdagradDOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kStackOpName).set_backend_op_name(kPackOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kUnstackOpName).set_backend_op_name(kUnpackOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kApplyAdagradDAOpName)
  .set_target_op_name(kApplyAdagradDADOpName)
  .set_need_tbe_check_supported(true);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kAdamOpName)
  .set_backend_op_name(kApplyAdamOpName)
  .set_target_op_name(kApplyAdamDOpName)
  .set_need_tbe_check_supported(true);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kApplyAdagradV2OpName).set_backend_op_name(kApplyAdagradV2DOpName);
REG_ASCEND_VM_OP_ADAPTATION_INFO(kApplyAdamWithAmsgradOpName)
  .set_target_op_name(kApplyAdamWithAmsgradDOpName)
  .set_input_attr_info(7, "float")
  .set_input_attr_info(8, "float")
  .set_input_attr_info(9, "float");
REG_ASCEND_VM_OP_ADAPTATION_INFO(kApplyRMSPropOpName)
  .set_target_op_name(kApplyRMSPropDOpname)
  .set_input_attr_info(5, "float")
  .set_input_attr_info(6, "float")
  .set_input_attr_info(7, "float");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kArgmaxOpName).set_backend_op_name(kArgMaxDOpName);

REG_ASCEND_VM_OP_ADAPTATION_INFO(kArgMaxV2OpName)
  .set_target_op_name(kArgMaxDOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(1, "int");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kArgminOpName)
  .set_backend_op_name(kArgMinOpName)
  .set_target_op_name(kArgMinDOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(1, "int");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kArgminV2OpName).set_backend_op_name(kArgMinOpName);

REG_ASCEND_VM_OP_ADAPTATION_INFO(kAvgPoolGradOpName)
  .set_target_op_name(kAvgPoolGradDOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(0, "listInt");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kAvgPoolGradVmOpName)
  .set_target_op_name(kAvgPoolGradDOpName)
  .set_input_attr_info(0, "listInt");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kAvgPool3DGradOpName)
  .set_target_op_name(kAvgPool3DGradDOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(0, "listInt");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kBatchToSpaceOpName)
  .set_target_op_name(kBatchToSpaceDOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(1, "listInt");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kBatchToSpaceNDOpName).set_backend_op_name(kBatchToSpaceNDDOpName);

REG_ASCEND_VM_OP_ADAPTATION_INFO(kCastOpName).set_target_op_name(kCastOpName).set_input_attr_info(1, "int");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kCentralizationOpName)
  .set_target_op_name(kCentralizationOpName)
  .set_input_attr_info(1, "listInt");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kClipBoxesOpName)
  .set_target_op_name(kClipBoxesDOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(1, "listInt");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kConcatOpName).set_backend_op_name(kConcatDOpName);

// index mismatch 2 vs 0 in cann
REG_ASCEND_VM_OP_ADAPTATION_INFO(kConv2DBackpropFilterOpName)
  .set_target_op_name(kConv2DBackpropFilterDOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(2, "listInt");

// index mismatch 2 vs 0 in cann
REG_ASCEND_VM_OP_ADAPTATION_INFO(kConv2DBackpropInputOpName)
  .set_target_op_name(kConv2DBackpropInputDOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(2, "listInt");

// index mismatch 2 vs 0 in cann
REG_ASCEND_VM_OP_ADAPTATION_INFO(kConv2DTransposeOpName)
  .set_target_op_name(kConv2DTransposeDOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(2, "listInt");

// index mismatch 2 vs 0 in cann
REG_ASCEND_VM_OP_ADAPTATION_INFO(kConv3DBackpropFilterOpName)
  .set_target_op_name(kConv3DBackpropFilterDOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(2, "listInt");

// index mismatch 2 vs 0 in cann
REG_ASCEND_VM_OP_ADAPTATION_INFO(kConv3DBackpropInputOpName)
  .set_target_op_name(kConv3DBackpropInputDOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(2, "listInt");

// index mismatch 2 vs 0 in cann
REG_ASCEND_VM_OP_ADAPTATION_INFO(kConv3DTransposeOpName)
  .set_target_op_name(kConv3DTransposeDOpName)
  .set_input_attr_info(2, "listInt");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kCropAndResizeOpName)
  .set_target_op_name(kCropAndResizeDOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(3, "listInt");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kCumProdOpName)
  .set_backend_op_name(kCumprodOpName)
  .set_target_op_name(kCumprodDOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(1, "int");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kCumSumOpName)
  .set_backend_op_name(kCumsumOpName)
  .set_target_op_name(kCumsumDOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(1, "int");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kCumulativeLogsumexpOpName)
  .set_target_op_name(kCumulativeLogsumexpDOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(1, "int");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kDepthwiseConv2dNativeBackpropFilterOpName)
  .set_backend_op_name(kDepthwiseConv2DBackpropFilterOpName)
  .set_target_op_name(kDepthwiseConv2DBackpropDFilterOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(1, "listInt");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kDepthwiseConv2dNativeBackpropInputOpName)
  .set_backend_op_name(kDepthwiseConv2DBackpropInputOpName)
  .set_target_op_name(kDepthwiseConv2DBackpropInputDOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(0, "listInt");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kDropoutDoMaskV3OpName)
  .set_backend_op_name(kDropOutDoMaskV3OpName)
  .set_target_op_name(kDropOutDoMaskV3DOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(2, "float");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kEuclideanNormOpName)
  .set_target_op_name(kEuclideanNormDOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(1, "listInt");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kExpandDimsOpName).set_target_op_name(kExpandDimsOpName).set_input_attr_info(1, "int");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kFillOpName).set_target_op_name(kFillDOpName);

REG_ASCEND_VM_OP_ADAPTATION_INFO(kFillV2OpName).set_backend_op_name(kFillOpName).set_need_tbe_check_supported(true);

// In hisi code, first check dynamic impl in GatherV2
REG_ASCEND_VM_OP_ADAPTATION_INFO(kGatherOpName).set_backend_op_name(kGatherV2OpName);

REG_ASCEND_VM_OP_ADAPTATION_INFO(kSparseGatherV2OpName)
  .set_backend_op_name(kGatherV2OpName)
  .set_target_op_name(kGatherV2DOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(2, "int");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kHistogramFixedWidthOpName)
  .set_target_op_name(kHistogramFixedWidthDOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(2, "int");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kInTopKOpName).set_backend_op_name(kInTopKDOpName);

REG_ASCEND_VM_OP_ADAPTATION_INFO(kMaxPoolV2OpName)
  .set_target_op_name(kMaxPoolExt2OpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(1, "listInt")
  .set_input_attr_info(2, "listInt");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kOneHotOpName)
  .set_target_op_name(kOneHotDOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(1, "int");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kPadOpName).set_target_op_name(kPadDOpName).set_input_attr_info(1, "listListInt");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kReduceAllOpName)
  .set_target_op_name(kReduceAllDOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(1, "listInt");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kReduceAnyOpName)
  .set_target_op_name(kReduceAnyDOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(1, "listInt");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kReduceMaxOpName)
  .set_target_op_name(kReduceMaxDOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(1, "listInt");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kReduceMeanOpName)
  .set_target_op_name(kReduceMeanDOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(1, "listInt");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kReduceMinOpName)
  .set_target_op_name(kReduceMinDOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(1, "listInt");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kReduceProdOpName)
  .set_target_op_name(kReduceProdDOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(1, "listInt");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kReduceSumOpName)
  .set_target_op_name(kReduceSumDOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(1, "listInt");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kResizeBilinearV2OpName)
  .set_target_op_name(kResizeBilinearV2DOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(1, "listInt");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kResizeNearestNeighborGradOpName)
  .set_backend_op_name(kResizeNearestNeighborV2GradOpName)
  .set_target_op_name(kResizeNearestNeighborV2GradDOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(1, "listInt");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kResizeNearestNeighborOpName).set_backend_op_name(kResizeNearestNeighborV2DOpName);

REG_ASCEND_VM_OP_ADAPTATION_INFO(kReverseV2OpName)
  .set_target_op_name(kReverseV2DOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(1, "listInt");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kRpnProposalsOpName)
  .set_target_op_name(kRpnProposalsDOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(1, "listInt");

// index mismatch 2 vs 1 in cann
REG_ASCEND_VM_OP_ADAPTATION_INFO(kScatterNdOpName)
  .set_target_op_name(kScatterNdDOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(2, "listInt");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kSpaceToBatchOpName)
  .set_target_op_name(kSpaceToBatchDOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(1, "listInt");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kSpaceToBatchNDOpName)
  .set_target_op_name(kSpaceToBatchNDDOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(1, "listInt")
  .set_input_attr_info(2, "listInt");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kSparseApplyAdadeltaOpName)
  .set_target_op_name(kSparseApplyAdadeltaDOpName)
  .set_input_attr_info(5, "float");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kSparseApplyAdagradV2OpName).set_backend_op_name(kSparseApplyAdagradV2DOpName);

REG_ASCEND_VM_OP_ADAPTATION_INFO(kSparseApplyFtrlOpName).set_backend_op_name(kSparseApplyFtrlDOpName);

REG_ASCEND_VM_OP_ADAPTATION_INFO(kSparseApplyFtrlV2OpName).set_backend_op_name(kSparseApplyFtrlV2DOpName);

REG_ASCEND_VM_OP_ADAPTATION_INFO(kSparseApplyRMSPropOpName)
  .set_target_op_name(kSparseApplyRMSPropDOpName)
  .set_input_attr_info(4, "float")
  .set_input_attr_info(5, "float")
  .set_input_attr_info(6, "float");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kSplitOpName).set_backend_op_name(kSplitDOpName);

REG_ASCEND_VM_OP_ADAPTATION_INFO(kSplitVOpName).set_backend_op_name(kSplitVDOpName);

REG_ASCEND_VM_OP_ADAPTATION_INFO(kStridedSliceAssignOpName)
  .set_target_op_name(kStridedSliceAssignDOpName)
  .set_input_attr_info(1, "listInt")
  .set_input_attr_info(2, "listInt")
  .set_input_attr_info(3, "listInt");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kStridedSliceOpName)
  .set_target_op_name(kStridedSliceDOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(1, "listInt")
  .set_input_attr_info(2, "listInt")
  .set_input_attr_info(3, "listInt");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kTileOpName)
  .set_target_op_name(kTileDOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(1, "listInt");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kTransposeOpName)
  .set_target_op_name(kTransposeDOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(1, "listInt");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kUnsortedSegmentMaxOpName)
  .set_target_op_name(kUnsortedSegmentMaxDOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(2, "int");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kUnsortedSegmentMinOpName)
  .set_target_op_name(kUnsortedSegmentMinDOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(2, "int");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kUnsortedSegmentProdOpName)
  .set_target_op_name(kUnsortedSegmentProdDOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(2, "int");

REG_ASCEND_VM_OP_ADAPTATION_INFO(kUnsortedSegmentSumOpName)
  .set_target_op_name(kUnsortedSegmentSumDOpName)
  .set_need_tbe_check_supported(true)
  .set_input_attr_info(2, "int");
}  // namespace mindspore::opt

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_MINDIR_REG_ASCEND_VM_OP_ADAPTATION_INFO_H_
