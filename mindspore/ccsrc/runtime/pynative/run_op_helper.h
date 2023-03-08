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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_RUN_OP_RUN_OP_HELPER_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_RUN_OP_RUN_OP_HELPER_H_

#include <vector>
#include "include/backend/kernel_graph.h"
#include "runtime/hardware/device_context.h"

namespace mindspore::runtime {
// Update Tensor or input node DeviceAddress before PyNative async running.
void UpdateDeviceAddress(const KernelGraphPtr &graph, const std::vector<tensor::TensorPtr> &tensors_without_value_mask,
                         const device::DeviceContext *device_context);

void RunSingleOpGraph(const KernelGraphPtr &graph, const std::vector<tensor::TensorPtr> &input_tensors,
                      const device::DeviceContext *device_context);
void RunSingleOpGraphDynamic(const KernelGraphPtr &graph, const std::vector<tensor::TensorPtr> &input_tensors,
                             const device::DeviceContext *device_context, bool is_gradient_out);
}  // namespace mindspore::runtime
#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_RUN_OP_RUN_OP_HELPER_H_
