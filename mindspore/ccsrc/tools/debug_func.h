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

#ifndef MINDSPORE_CCSRC_TOOLS_DEBUG_FUNC_H_
#define MINDSPORE_CCSRC_TOOLS_DEBUG_FUNC_H_

#include <vector>
#include "include/backend/common/kernel_graph/kernel_graph.h"
#include "include/utils/anfalgo.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"

namespace mindspore {

namespace tools {

using device::DeviceContext;

void DebugOnStepBegin(const std::vector<KernelGraphPtr> &graphs, const std::vector<AnfNodePtr> &origin_parameters_order,
                      std::vector<DeviceContext *> device_contexts);

void DebugPostLaunch(const AnfNodePtr &node, const std::vector<kernel::KernelTensorPtr> &input_kernel_tensors,
                     const std::vector<kernel::KernelTensorPtr> &output_kernel_tensors,
                     const DeviceContext *device_context);
void DebugOnStepEnd(int total_running_count, std::vector<const DeviceContext *> device_contexts);

void DebugFinalize();

}  // namespace tools
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_TOOLS_DEBUG_FUNC_H_
