/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "runtime/device/ascend/executor/hccl_dynamic_kernel.h"
#include "runtime/device/ascend/executor/rts/memcpy_rts_dynamic_kernel.h"
#include "runtime/device/ascend/executor/rts/profiling_rts_dynamic_kernel.h"
#include "runtime/device/ascend/executor/ai_core_dynamic_kernel.h"
#include "profiler/device/ascend/rt_callback_manager.h"
#include "profiler/device/ascend/ascend_profiling.h"
#include "runtime/device/ascend/executor/tiling/op_tiling_calculater.h"
#include "backend/kernel_compiler/host/host_kernel_metadata.h"
#include "backend/kernel_compiler/host/host_kernel_build.h"

namespace mindspore {
namespace device {
namespace ascend {
void HcclDynamicKernel::UpdateArgs() {}
void HcclDynamicKernel::Execute() {}
void HcclDynamicKernel::PostExecute() {}

void MemcpyRtsDynamicKernel::Execute() {}

void ProfilingRtsDynamicKernel::Execute() {}

AiCoreDynamicKernel::~AiCoreDynamicKernel() {}
void AiCoreDynamicKernel::Execute() {}
void AiCoreDynamicKernel::UpdateArgs() {}
void AiCoreDynamicKernel::Initialize() {}
void AiCoreDynamicKernel::PostExecute() {}

bool HcclExecutorManager::Initialize() { return true; }
bool HcclExecutorManager::Finalize() { return true; }

void OpTilingCalculater::Init() {}
void OpTilingCalculater::CalculateTiling(const NotNull<CNodePtr> &cnode, const optiling::OpCompileInfo &op_compile_info,
                     const std::map<uint32_t, tensor::TensorPtr> &depend_tensor_map,
                     NotNull<optiling::OpRunInfo *> op_run_info) {}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

namespace mindspore {
namespace profiler {
namespace ascend {
CallbackManager::CallbackManager(rtStream_t stream) : stream_(stream) {}
Status CallbackManager::Init() { return kSuccess; }
Status CallbackManager::Destroy() { return kSuccess; }
Status CallbackManager::RegisterCallback(rtCallback_t callback, const void *user_data) { return kSuccess; }
Status CallbackManager::RegisterCallback(const std::function<void()> &callback) { return kSuccess; }

AscendProfiler::AscendProfiler() : counter_(0) { Reset(); }

void AscendProfiler::RecordEvent(EventType event_type, const char *fmt, ...) {}

void AscendProfiler::Dump(std::ostream &output_stream) {}

void AscendProfiler::Reset() {}
}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore

namespace mindspore {
namespace kernel {
void HostMetadataInfo(const CNodePtr &kernel_node, std::vector<std::shared_ptr<KernelBuildInfo>> *kernel_info_list) {}
KernelModPtr HostOpBuild(const std::shared_ptr<AnfNode> &anf_node) { return nullptr; }
}  // namespace kernel
}  // namespace mindspore
