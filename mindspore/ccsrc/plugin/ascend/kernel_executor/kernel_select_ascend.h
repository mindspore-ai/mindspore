/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include <utility>
#include <string>
#include <tuple>
#include <vector>
#include "ir/anf.h"
#include "utils/ms_context.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel_build_info.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel_info.h"
#include "include/backend/common/kernel_graph/kernel_graph.h"

namespace mindspore {
namespace device {
namespace ascend {
enum SelectedKernelType {
  KERNEL_TYPE_BEGIN,
  INTERNAL_KERNEL = KERNEL_TYPE_BEGIN,
  ACLNN_KERNEL,
  ACLOP_KERNEL,
  ATB_KERNEL,
  HCCL_KERNEL,
  HOST_KERNEL,
  CUSTOM_KERNEL,
  KERNEL_TYPE_END,
  NUM_KERNLE_TYPE = KERNEL_TYPE_END - KERNEL_TYPE_BEGIN
};

void HandleKernelSelectFailure(const KernelGraphPtr &graph, const CNodePtr &node,
                               const std::pair<std::string, ExceptionType> &failure_info);

std::tuple<bool, std::string, ExceptionType, bool> SelectKernelInfoWithMsg(
  const KernelGraphPtr &graph, const CNodePtr &node, std::vector<size_t> *op_selected_num = nullptr);

bool IsEnableAclnn(const KernelGraphPtr &kernel_graph, const CNodePtr &node);

void SetKernelInfoBeforeCreateKernel(const std::vector<CNodePtr> &nodes);

void GenerateKernelBuildInfo(const CNodePtr &kernel, const KernelType &kernel_type);
BACKEND_EXPORT bool IsEmptyTupleInput(const CNodePtr &kernel, const size_t i, const TypeId cur_type_id);
BACKEND_EXPORT TypeId GetInputDeviceType(const AnfNodePtr &kernel_node, size_t input_idx);
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
