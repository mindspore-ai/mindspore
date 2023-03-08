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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_KERNEL_SELECT_ASCEND_ANFALGO_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_KERNEL_SELECT_ASCEND_ANFALGO_H_
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
enum KernelSelectStatus {
  kNoMatched = -1,
  kStatusAllMatched = 0,
  kStatusReducePrecision = 1,
  kStatusRaisePrecision = 2,
};
KernelSelectStatus SelectKernelInfo(const CNodePtr &kernel_node,
                                    KernelType kernel_type = KernelType::UNKNOWN_KERNEL_TYPE);
std::tuple<KernelSelectStatus, std::string, ExceptionType> SelectKernelInfoWithMsg(
  const CNodePtr &kernel_node, KernelType kernel_type = KernelType::UNKNOWN_KERNEL_TYPE);
void SetTensorDeviceInfo(const CNodePtr &kernel_node);
void SelectGraphKernelInfo(const CNodePtr &kernel_node, const FuncGraphPtr &func_graph);
void SetAscendKernelInfo(const CNodePtr &kernel_node, KernelType kernel_type);
// After operator selection in graph optimization, new nodes will be added, select kernel info for those nodes
// check whether the node has completed the operator selection. If not, the operator
// selection needs to be performed to set kernel info.
void SelectKernelInfoAfterKernelSelect(const std::vector<CNodePtr> &nodes);
// Mark the kernel backoff with failure info when setting operator info fails.
void HandleKernelSelectFailure(const KernelGraphPtr &graph, const CNodePtr &node,
                               const std::pair<std::string, ExceptionType> &failure_info);

class AscendGraphKernelInfo : public GraphKernelInfo {
 public:
  AscendGraphKernelInfo() = default;
  virtual ~AscendGraphKernelInfo() = default;
  void SetKernelInfo(const CNodePtr &kernel_node, KernelType kernel_type) override {
#ifndef ENABLE_ACL
    SetAscendKernelInfo(kernel_node, kernel_type);
#endif
  }
};

REG_GRAPH_KERNEL_INFO(kAscendDevice, AscendGraphKernelInfo);
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_KERNEL_SELECT_ASCEND_ANFALGO_H_
