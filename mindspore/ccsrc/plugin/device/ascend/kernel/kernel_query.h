/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_KERNEL_QUERY_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_KERNEL_QUERY_H_

#include <vector>
#include <memory>
#include "kernel/kernel.h"
#include "kernel/kernel_build_info.h"

namespace mindspore {
namespace kernel {
void KernelQuery(const CNodePtr &kernel_node, std::vector<std::shared_ptr<kernel::KernelBuildInfo>> *kernel_info_list,
                 KernelType kernel_type = KernelType::UNKNOWN_KERNEL_TYPE);
void AICPUQuery(const CNodePtr &kernel_node, std::vector<std::shared_ptr<kernel::KernelBuildInfo>> *kernel_info_list);
bool IsSupportedByAICPU(const AnfNodePtr &kernel_node, const KernelBuildInfoPtr &select_kernel_build_info);
void BishengQuery(const CNodePtr &kernel_node, std::vector<std::shared_ptr<kernel::KernelBuildInfo>> *kernel_info_list);
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_KERNEL_QUERY_H_
