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
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

namespace mindspore {
namespace kernel {
void HostMetadataInfo(const CNodePtr &kernel_node, std::vector<std::shared_ptr<KernelBuildInfo>> *kernel_info_list) {}
KernelModPtr HostOpBuild(const std::shared_ptr<AnfNode> &anf_node) { return nullptr; }
}  // namespace kernel
}  // namespace mindspore
