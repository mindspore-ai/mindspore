/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_GE_BACKEND_EXECUTOR_GE_MEMORY_ALLOCATOR_H_
#define MINDSPORE_CCSRC_BACKEND_GE_BACKEND_EXECUTOR_GE_MEMORY_ALLOCATOR_H_

#include <memory>
#include <string>
#include <set>
#include "include/runtime/hardware_abstract/kernel_base/kernel_tensor.h"
#include "include/backend/common/kernel_graph/kernel_graph.h"
#include "backend/ge_backend/graph_ir/types.h"
#include "backend/ge_backend/executor/ge_device_res_manager.h"
#include "backend/ge_backend/executor/ge_summary.h"

namespace mindspore {
namespace backend {
namespace ge_backend {
class GEMemoryAllocator {
 public:
  static void ProcessGraphDeviceAddress(const KernelGraphPtr &kernel_graph, GeDeviceResManagerPtr res_manager);
  static void AllocGraphMemory(const backend::ge_backend::RunOptions &options, const KernelGraphPtr &graph,
                               const GraphSummary &summary, size_t stream_id, GeDeviceResManagerPtr res_manager);
  static void AllocUnuseInput(const KernelGraphPtr &kernel_graph, const AnfNodePtr &input_node,
                              device::DeviceAddress *output_addr, GeDeviceResManagerPtr res_manager);
  static void AllocUnuseInput(const KernelGraphPtr &kernel_graph, kernel::KernelTensor *tensor,
                              GeDeviceResManagerPtr res_manager);
};
}  // namespace ge_backend
}  // namespace backend
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_GE_BACKEND_EXECUTOR_GE_MEMORY_ALLOCATOR_H_
