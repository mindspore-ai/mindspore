/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_RUNTIME_UTILS_RUNTIME_CONF_RUNTIME_ENV_H_
#define MINDSPORE_CCSRC_INCLUDE_RUNTIME_UTILS_RUNTIME_CONF_RUNTIME_ENV_H_

#include <memory>
#include <utility>
#include <string>
#include "runtime/utils/visible.h"

namespace mindspore {
namespace runtime {
// Runtime dev config.
const char kRuntimeConf[] = "MS_DEV_RUNTIME_CONF";
const char kRuntimeInline[] = "inline";
const char kRuntimeSwitchInline[] = "switch_inline";
const char kRuntimeNewRefCount[] = "new_ref_count";
const char kRuntimeMultiStream[] = "multi_stream";
constexpr char kRuntimeMc2Event[] = "mc2_event";
const char kRuntimePipeline[] = "pipeline";
const char kRuntimeNewPipeline[] = "new_pipeline";
const char kRuntimeGraphPipeline[] = "graph_pipeline";
const char kRuntimeKbkSubGraphMode[] = "kbk_sub_graph_mode";
const char kRuntimeCommunicationLaunchGroup[] = "communication_launch_group";
const char kRuntimeInsertTensorMove[] = "insert_tensormove";
const char kRuntimeAllfinite[] = "all_finite";
const char kRuntimeGeKernel[] = "ge_kernel";
const char kRuntimeCache[] = "backend_compile_cache";
const char kRuntimeCopyAsync[] = "copy_async";
const char kRuntimeSyncCopyInput[] = "sync_copy_input";
const char kRuntimeClusterThreadNum[] = "cluster_thread_num";
const char kRuntimeThreadLoadCache[] = "multi_thread_load_cache";
const char kRuntimeAsyncInitComm[] = "async_init_comm";
const char kRuntimeCpuAffinityList[] = "cpu_affinity_list";
const char kRuntimeCpuAffinityMoudule[] = "cpu_affinity_module";
const char kRuntimeActorThreadFixBind[] = "actor_thread_fix_bind";
const char kRuntimeInputOptimize[] = "input_optimize";
const char kRuntimeCommInitLcclOnly[] = "comm_init_lccl_only";
const char kRuntimeGraphOrder[] = "graph_order";
const char kRuntimeSyncStreamOnDemand[] = "sync_stream_on_demand";
const char kRuntimeGraphCaptureMaxNumber[] = "graph_capture_max_number";
// Runtime debug config.
const char kRuntimeMemoryTrack[] = "memory_track";
const char kRuntimeMemoryStat[] = "memory_statistics";
const char kRuntimeCompileStat[] = "compile_statistics";
const char kRuntimeAclnnCacheQueueLength[] = "aclnn_cache_queue_length";
const char kRuntimePreBuildCommKernel[] = "pre_build_comm_kernel";
const char kRuntimeExecutionOrderCheckIteration[] = "execution_order_check_iteration";
const char kRuntimeHPMode[] = "high_performance_mode";

RUNTIME_UTILS_EXPORT std::string GetRuntimeConfigValue(const std::string &runtime_config);
RUNTIME_UTILS_EXPORT bool IsEnableRuntimeConfig(const std::string &runtime_config);
RUNTIME_UTILS_EXPORT bool IsDisableRuntimeConfig(const std::string &runtime_config);
}  // namespace runtime
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_RUNTIME_UTILS_RUNTIME_CONF_RUNTIME_ENV_H_
