/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_COMMON_COMMON_H_
#define MINDSPORE_LITE_SRC_COMMON_COMMON_H_

#include <string>
#include "mindspore/ccsrc/plugin/device/cpu/kernel/nnacl/op_base.h"

/*
Naming a key of path must be consistent with existing naming styles and follow the following rules:
(1) If a path points to a file, please name it xxx_file;
(2) If a path points to a dir, please name it xxx_dir;
(3) If others, please make a technical review.
*/

namespace mindspore {
namespace lite {
enum NCHW_SHAPE { NCHW_N = 0, NCHW_C = 1, NCHW_H = 2, NCHW_W = 3 };
enum NHWC_SHAPE { NHWC_N = 0, NHWC_H = 1, NHWC_W = 2, NHWC_C = 3 };
enum HWCK_SHAPE { HWCK_H = 0, HWCK_W = 1, HWCK_C = 2, HWCK_K = 3 };
enum HWKC_SHAPE { HWKC_H = 0, HWKC_W = 1, HWKC_K = 2, HWKC_C = 3 };
enum KCHW_SHAPE { KCHW_K = 0, KCHW_C = 1, KCHW_H = 2, KCHW_W = 3 };
enum CKHW_SHAPE { CKHW_C = 0, CKHW_K = 1, CKHW_H = 2, CKHW_W = 3 };
enum CHWK_SHAPE { CHWK_C = 0, CHWK_H = 1, CHWK_W = 2, CHWK_K = 3 };
enum KHWC_SHAPE { KHWC_K = 0, KHWC_H = 1, KHWC_W = 2, KHWC_C = 3 };
enum CHW_SHAPE { CHW_C = 0, CHW_H = 1, CHW_W = 2 };
enum HWC_SHAPE { HWC_H = 0, HWC_W = 1, HWC_C = 2 };
static constexpr int kHWDimNumber = 2;
static constexpr int kCHWDimNumber = 3;
static constexpr int kNCHWDimNumber = 4;
static constexpr int kNHWCDimNumber = 4;

static constexpr int TENSOR_MAX_REFCOUNT = 999;

// quantization relative
static const char QUANTIZED_UINT8[] = "QUANTIZED_UINT8";
static const char QUANTIZED_INT8[] = "QUANTIZED_INT8";
static const char QUANTIZED_INT16[] = "QUANTIZED_INT16";
static const char QUANTIZED_UINT16[] = "QUANTIZED_UINT16";
static const char QUANTIZED_FLOAT16[] = "FLOAT16";
static const char QUANTIZED_FLOAT32[] = "FLOAT32";
static const char QUANTIZATION_TYPE_DYNAMIC[] = "DYNAMIC";
static const char QUANTIZATION_TYPE_STATIC[] = "STATIC";
static const char CALIB_NORM[] = "NORM";

// dims
static const int32_t DIM_DEFAULT_SIZE = 4;
static const char *const kIsOptimized = "isOptimized";
static const char *const kInputFormat = "inputFormat";
static const char *const kOutputFormat = "outputFormat";
static const char *const kIsDynamicShape = "isDynamicShape";
// ms cache
static const char *const kMSCacheSection = "ms_cache";
static const char *const kMSCacheModelPathKey = "cache_model_path";
static const char *const kMSCacheVocabSizeKey = "vocab_size";
static const char *const kMSCacheDeviceSizeKey = "device_cache_size";
static const char *const kMSCacheSerializePathKey = "serialize_path";
// mindir weight path
static const char *const kConfigModelFileSection = "model_file";
static const char *const kConfigMindIRPathKey = "mindir_path";
static const char *const kWeightSection = "weight";
static const char *const kWeightPathKey = "weight_path";
// shared parallel thread pool
static const char *const kSharedThreadPoolSection = "shared_thread_pool";
static const char *const kEnableSharedThreadPoolKey = "enable_shared_thread_pool";
static const char *const kThreadNumLimitPerWorkerKey = "thread_num_limit_per_worker";
static const char *const kThreadNumRemainingPerWorkerKey = "thread_num_remaining_per_worker";
// model pool inner section and key
static const char *const kInnerModelParallelRunnerSection = "inner_model_parallel_runner";
static const char *const kInnerSharingWeightCopyBufKey = "sharing_weight_copy_buf";
static const char *const kInnerModelIDKey = "inner_model_id";
static const char *const kInnerRunnerIDKey = "inner_runner_id";
static const char *const kInnerNumaIDKey = "inner_numa_id";
static const char *const kInnerWorkerNumKey = "inner_worker_num";

// common context
static const char *const kCommonContextSection = "common_context";
// gpu context
static const char *const kGPUContextSection = "gpu_context";
static const char *const kInputShapeKey = "input_shape";
static const char *const kDynamicDimsSearchKey = "ge.dynamicDims";
static const char *const kDynamicDimsKey = "dynamic_dims";
static const char *const kOptimizeDimsKey = "opt_dims";
static const char *const kPrecisionModeKey = "precision_mode";
static const char *const kDumpOpsKey = "dump_ops";
static const char *const kDumpDirKey = "dump_dir";
// ascend context
static const char *const kAscendContextSection = "ascend_context";
static const char *const kProfilingPathKey = "profiling_config_file";
static const char *const kDumpPathKey = "dump_config_file";
static const char *const kDumpModelNameKey = "dump_model_name";
static const char *const kGeVariableMemoryMaxSize = "ge_variable_memory_max_size";
static const char *const kGeGraphMemoryMaxSize = "ge_graph_memory_max_size";
static const char *const kGraphCompilerCacheDirKey = "model_cache_dir";
static const char *const kModifyMixList = "mixprecision_list_path";
static const char *const kEnableCustomOp = "enable_custom_op";
static const char *const kPluginCustomOps = "plugin_custom_ops";
static const char *const kOpAttrs = "op_attrs";
static const char *const kAoeMode = "aoe_mode";
static const char *const kProvider = "provider";
static const char *const kAscendProviderGe = "ge";
static const char *const kParameterAsRefData = "parameter_as_refdata";
static const char *const kModelCacheMode = "model_cache_mode";
// ge options
static const char *const kGeSessionOptionsSection = "ge_session_options";
static const char *const kGeGraphOptionsSection = "ge_graph_options";
static const char *const kGeGlobalOptionsSection = "ge_global_options";
// distributed infer
static const char *const kRankTableFilePathKey = "rank_table_file";
// transformer
static const char *const kTransformerSection = "transformer_context";
static const char *const kEncoderInputKey = "encoder_input";
static const char *const kDecoderInputKey = "decoder_input";
static const char *const kFfnFp16Key = "ffn_fp16";
static const char *const kOptimizeTransformer = "optimize_transformer";
// lite inner group
static const char *const kLiteInnerGroupSection = "lite_inner_group";
static const char *const kLiteInnerGroupId = "group_id";
// aoe section
static const char *const kAoeGlobalOptionsSection = "aoe_global_options";
static const char *const kAoeTuningOptionsSection = "aoe_tuning_options";
// acl options
static const char *const kAclOptionParam = "acl_option_cfg_param";
static const char *const kAclInitOptionParam = "acl_init_options";
static const char *const kAclBuildOptionParam = "acl_build_options";

static const char *const kNameAttrWeightDir = "weight_dir";

static const char *const kOutputShapes = "outputs_shape";
static const char *const kNameCustomAscend = "CustomAscend";

static const char *const kNameAttrRefDatas = "ref_datas";
static const char *const kNameAttrZeroValRefDatas = "zero_val_ref_datas";
static const char *const kConverterParams = "converter_params";
static const char *const kConverterOutputFile = "output_file";
// om converter options
static const char *const kOMConverterOptionsSection = "om_converter";

static const char *const KModelUserInfo = "user_info";
static const char *const KModelInputShape = "input_shape";

// for [ascend_context] [kvcache_input_layout]
static const char *const kKVCacheLayoutBNSD = "BNSD";
static const char *const kKVCacheLayoutBSH = "BSH";

// for sharing mem
static const char *const kInnerCommon = "inner_common";
static const char *const kInnerSharingWorkspace = "inner_sharing_workspace";
static const char *const kInnerModelPath = "inner_model_path";
static const char *const kInnerWeightspace = "inner_weightspace";
static const char *const kInnerWorkspace = "inner_workspace";
static const char *const kInnerCalcWorkspaceSize = "inner_calc_workspace_size";
static const char *const kInnerWeightspaceWorkspace = "inner_weightspace_workspace";
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_COMMON_COMMON_H_
