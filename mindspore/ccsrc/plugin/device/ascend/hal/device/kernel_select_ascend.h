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
#include "kernel/kernel_build_info.h"
#include "kernel/graph_kernel_info.h"
#include "include/backend/kernel_graph.h"

namespace mindspore {
namespace device {
namespace ascend {
void HandleKernelSelectFailure(const KernelGraphPtr &graph, const CNodePtr &node,
                               const std::pair<std::string, ExceptionType> &failure_info);
std::tuple<bool, std::string, ExceptionType> SelectKernelInfoWithMsg(const CNodePtr &node, bool enable_aclnn = false);
void SetKernelInfoBeforeCreateKernel(const std::vector<CNodePtr> &nodes);
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
