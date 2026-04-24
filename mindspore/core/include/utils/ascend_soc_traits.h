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

#ifndef MINDSPORE_CORE_INCLUDE_UTILS_ASCEND_SOC_TRAITS_H_
#define MINDSPORE_CORE_INCLUDE_UTILS_ASCEND_SOC_TRAITS_H_

#include <string>
#include "mindapi/base/macros.h"

namespace mindspore {
namespace ascend_soc {

// Normalized SOC strings come from MsContext::ascend_soc_version() after ascend_device_context mapping.
// Feature names are stable API keys; supported SOCs per feature are defined only in ascend_soc_traits.cc.

// --- ascend910 classic training line: one capability per call site (do not reuse across different semantics) ---
constexpr char kAscendFeatureJitLevelO2DefaultAscendJit[] = "jit_level_o2_default_ascend_jit";
constexpr char kAscendFeatureRepeatInterleaveOpTrainClassicConstraint[] =
  "repeat_interleave_op_train_classic_constraint";
constexpr char kAscendFeatureGraphKernelStitchFusionDefaultTrainClassic[] =
  "graph_kernel_stitch_fusion_default_train_classic";
constexpr char kAscendFeatureParallelDeviceListCheckTrainClassic[] = "parallel_device_list_check_train_classic";
constexpr char kAscendFeatureRuntimeDisableGeKernelTrainClassic[] = "runtime_disable_ge_kernel_train_classic";
constexpr char kAscendFeatureGeFormatModeGraphSinkDefaultTrainClassic[] =
  "ge_format_mode_graph_sink_default_train_classic";
constexpr char kAscendFeatureDeviceContextGeInitBeforeDistributedTrainClassic[] =
  "device_context_ge_init_before_distributed_train_classic";
constexpr char kAscendFeatureJitConfigBackendGeDefaultEmptyJit[] = "jit_config_backend_ge_default_empty_jit";
constexpr char kAscendFeaturePyQueryTrainClassicAcceleratorLine[] = "py_query_train_classic_accelerator_line";
constexpr char kAscendFeatureAmpLossScaleTrainClassicTarget[] = "amp_loss_scale_train_classic_target";

// --- ascend910b / ascend910_93 successor training line (split by call site) ---
constexpr char kAscendFeatureAclOverflowModeDeviceContextInitSuccessor[] =
  "acl_overflow_mode_device_context_init_successor";
constexpr char kAscendFeatureAclOverflowModeGeBackendInitSuccessor[] = "acl_overflow_mode_ge_backend_init_successor";
constexpr char kAscendFeatureInternalKernelNzFormatMatmulSuccessor[] = "internal_kernel_nz_format_matmul_successor";
constexpr char kAscendFeatureFusionMatmulElemwiseNonFastgeluSuccessor[] =
  "fusion_matmul_elemwise_non_fastgelu_successor";
constexpr char kAscendFeaturePyQueryTrainSuccessorFamilyLine[] = "py_query_train_successor_family_line";
constexpr char kAscendFeatureAvgPool2dPadConstraintTrainSuccessor[] = "avg_pool2d_pad_constraint_train_successor";

// GE / AOE dtype policy (same dtype rules, different pipelines — separate keys).
constexpr char kAscendFeatureGeSessionPrecisionMustKeepOriginDtypeCloud[] =
  "ge_session_ge_dtype_must_keep_src_dtype_plat";
constexpr char kAscendFeatureAoeOnlinePrecisionMustKeepOriginDtype[] = "aoe_online_ge_dtype_must_keep_src_dtype";

constexpr char kAscendFeatureParallelOverlapGradCommMainCloudTrain[] = "parallel_overlap_grad_comm_core_plat_train";
constexpr char kAscendFeatureParallelOverlapRecomputeAllgatherFaGradMainCloudTrain[] =
  "parallel_overlap_recompute_allgather_fa_grad_core_plat_train";

constexpr char kAscendFeatureGraphKernelDvm[] = "graph_kernel_dvm";
constexpr char kAscendFeatureFusionMatmulElemwiseMatmulElemwisePass[] = "fusion_matmul_elemwise_matmul_elemwise_pass";
constexpr char kAscendFeatureFusionMatmulElemwiseTransposeExtPass[] = "fusion_matmul_elemwise_transpose_ext_pass";
constexpr char kAscendFeatureInferBoost910bFamilyFusionMatmulSigmoidAdd[] =
  "infer_boost_910b_family_fusion_matmul_sigmoid_add";
constexpr char kAscendFeatureInferBoost910bFamilyFusionMatmulSigmoidCastAdd[] =
  "infer_boost_910b_family_fusion_matmul_sigmoid_cast_add";
constexpr char kAscendFeatureInferBoost910bFamilyFusionTransposeBatchMatmulTranspose[] =
  "infer_boost_910b_family_fusion_transpose_batch_matmul_transpose";
constexpr char kAscendFeatureInferBoost910bFamilyFusionSwigluDynamicQuant[] =
  "infer_boost_910b_family_fusion_swiglu_dynamic_quant";
constexpr char kAscendFeatureInferBoost910bFamilyFusionSwigluReshapeDynamicQuant[] =
  "infer_boost_910b_family_fusion_swiglu_reshape_dynamic_quant";
constexpr char kAscendFeatureInferBoost910bFamilyFusionMoeInitRoutingDynQuantV2[] =
  "infer_boost_910b_family_fusion_moe_init_routing_dyn_quantv2";
constexpr char kAscendFeatureInferBoostQbmmElemwise[] = "infer_boost_qbmm_elemwise";
constexpr char kAscendFeatureInternalKernelAclGraph[] = "internal_kernel_acl_graph";
constexpr char kAscendFeatureMc2Lccl[] = "mc2_lccl";
constexpr char kAscendFeatureMemoryDefault64gLine[] = "memory_default_64g_line";
constexpr char kAscendFeatureDeviceCapacity64gLine[] = "device_capacity_64g_line";
constexpr char kAscendFeatureAscendResManagerInfer310SkipsHcclTimeoutApis[] =
  "ascend_res_manager_infer310_skips_hccl_timeout_apis";
constexpr char kAscendFeaturePyQueryInfer310FamilyLine[] = "py_query_infer310_family_line";
constexpr char kAscendFeatureKbkTransData310pInternalKernel[] = "kbk_trans_data_310p_internal";
constexpr char kAscendFeatureKbkTransData910bNonCustom[] = "kbk_trans_data_910b_non_custom";

constexpr char kAscendFeatureGeExecFormatModePreferAltSocLine[] = "ge_exec_format_mode_prefer_alt_soc_line";
constexpr char kAscendFeatureShapeReshapeFusionNonGraphRunAltSocLine[] =
  "shape_reshape_fusion_non_graph_run_alt_soc_line";

constexpr char kAscendFeatureMemoryTotalInfer310pLine[] = "memory_total_infer_310p_line";
constexpr char kAscendFeatureMemoryTotalDevice950SkuTable[] = "memory_total_device_950_sku_table";
constexpr char kAscendFeatureInferCommSkipInterNodeHcclGroup[] = "infer_comm_skip_inter_node_hccl_group";
constexpr char kAscendFeatureCpuCollectiveAllreduceLauncherInfer310p[] = "cpu_collective_allreduce_starter_infer_310p";
constexpr char kAscendFeatureInternalKernelMatmulNzFormatInfer310p[] = "internal_kernel_matmul_nz_format_infer_310p";
constexpr char kAscendFeatureInferBoostFormatIdentityTransDataInternal[] =
  "infer_boost_format_identity_trans_data_internal";
constexpr char kAscendFeatureInferBoostGroupedMatmulInternalKernel310p[] =
  "infer_boost_grouped_matmul_internal_kernel_310p";
constexpr char kAscendFeatureInternalOpListDefaultInfer310p[] = "internal_op_list_default_infer_310p";

constexpr char kAscendFeatureMc2MatmulReduceScatterAclnnSupported[] = "mc2_matmul_reduce_scatter_aclnn_supported";
constexpr char kAscendFeatureMc2AllgatherMatmulAclnnSupported[] = "mc2_allgather_matmul_aclnn_supported";
constexpr char kAscendFeatureMc2Lccl91093Variant[] = "mc2_lccl_910_93_variant";

constexpr char kAscendFeatureDvmCommCollective910bNormalizedPath[] = "dvm_comm_collective_910b_normalized_path";
constexpr char kAscendFeatureMatmulAllreduceAddRmsnormFusion910bNormalizedLine[] =
  "matmul_allreduce_add_rmsnorm_fusion_910b_normalized_line";

constexpr char kAscendFeatureAtbInplaceGroupedMatmulSkipWorkspaceInit[] =
  "atb_inplace_grouped_matmul_skip_workspace_init";
constexpr char kAscendFeatureAtbInplaceGroupedMatmulRunOperatorLaunch[] = "atb_inplace_grouped_matmul_run_op_kernel";
constexpr char kAscendFeatureMindDataDvppSupportedSocLine[] = "minddata_dvpp_supported_soc_line";
constexpr char kAscendFeatureAoeAscend9ServerLine[] = "aoe_ascend9_server_line";
constexpr char kAscendFeatureMemoryVmmDefaultEnabledSocLine[] = "memory_vmm_default_enabled_soc_line";

// Mixed-dtype / hq (high-quality) dtype training line (loss scale, GE dtype policy, etc. — shared only where same
// policy).
constexpr char kAscendFeatureTrainHighPrecisionDtypeLine[] = "train_hq_dtype_line";

MS_CORE_API bool SocSupportsAscendFeatureForSocVersion(const std::string &soc_version, const std::string &feature_name);

MS_CORE_API bool SocSupportsAscendFeature(const std::string &feature_name);

}  // namespace ascend_soc
}  // namespace mindspore

#endif  // MINDSPORE_CORE_INCLUDE_UTILS_ASCEND_SOC_TRAITS_H_
