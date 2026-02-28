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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_HYPER_OFFLOAD_UTILS_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_HYPER_OFFLOAD_UTILS_H_

#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "backend/ms_backend/runtime/actors/base/kernel_runner.h"

namespace mindspore {
namespace runtime {

inline constexpr uint32_t kInvalidStreamId = std::numeric_limits<uint32_t>::max();
constexpr auto kSwitchTrueBranchNum = "TrueBranchNodeNum";
constexpr auto kSwitchFalseBranchNum = "FalseBranchNodeNum";
enum HyperOffloadEventType {
  kDeviceToHost,
  kHostToDevice,
  kRecordWithMemoryEvent,
  kWaitWithMemoryEvent,
  kRecordWaitPairEvent,
  kEventEnd,
};

/// \brief Runtime remote action metadata for copy/record/wait tasks.
struct RemoteAction {
  HyperOffloadEventType event_type;
  kernel::KernelTensorPtr kernel_tensor;
  uint32_t src_stream_id;
  uint32_t dst_stream_id;

  RemoteAction(HyperOffloadEventType type, kernel::KernelTensorPtr tensor = nullptr, uint32_t src = kInvalidStreamId,
               uint32_t dst = kInvalidStreamId)
      : event_type(type), kernel_tensor(std::move(tensor)), src_stream_id(src), dst_stream_id(dst) {}
};

/// \brief Condition switch branch index metadata used by HyperOffload horizon inference.
struct ConditionSwitchBranchInfo {
  CNodePtr switch_cnode = nullptr;
  size_t cur_idx{0};
  size_t true_branch_node_num{0};
  size_t false_branch_node_num{0};
  size_t start_true_idx{0};
  size_t end_true_idx{0};
  size_t start_false_idx{0};
  size_t end_false_idx{0};
  bool is_running_true_branch{true};

  ConditionSwitchBranchInfo(const CNodePtr &switch_cnode, size_t cur_idx)
      : switch_cnode(switch_cnode), cur_idx(cur_idx) {
    RefreshNodeIdx();
  }

  std::string ToString() const {
    std::ostringstream os;
    os << "Cur_idx: " << cur_idx << ", true branch idx range: " << start_true_idx << ", " << end_true_idx
       << ", false branch idx range: " << start_false_idx << ", " << end_false_idx;
    return os.str();
  }

  void RefreshNodeIdx() {
    MS_EXCEPTION_IF_NULL(switch_cnode);
    auto true_branch_node_num_value = switch_cnode->GetAttr(kSwitchTrueBranchNum);
    auto false_branch_node_num_value = switch_cnode->GetAttr(kSwitchFalseBranchNum);
    if (true_branch_node_num_value == nullptr || false_branch_node_num_value == nullptr) {
      MS_LOG(EXCEPTION) << "Invalid switch node, need branch node num info: " << switch_cnode->DebugString();
    }
    true_branch_node_num = GetValue<size_t>(true_branch_node_num_value);
    false_branch_node_num = GetValue<size_t>(false_branch_node_num_value);
    if (true_branch_node_num == 0 || false_branch_node_num == 0) {
      MS_LOG(EXCEPTION) << "Invalid switch node branch num for " << switch_cnode->DebugString()
                        << ", true_branch_node_num: " << true_branch_node_num
                        << ", false_branch_node_num: " << false_branch_node_num;
    }
    start_true_idx = cur_idx + 1;
    end_true_idx = start_true_idx + true_branch_node_num - 1;
    start_false_idx = end_true_idx + 1;
    end_false_idx = start_false_idx + false_branch_node_num - 1;
  }
};

using RemoteActionPtr = std::shared_ptr<RemoteAction>;
using RemoteActionPtrList = std::vector<RemoteActionPtr>;
using KernelTensorPtrList = std::vector<kernel::KernelTensorPtr>;
using KernelTensorPtrPair = std::pair<kernel::KernelTensorPtr, kernel::KernelTensorPtr>;
using KernelTensorPtrPairList = std::vector<KernelTensorPtrPair>;
using ConditionSwitchBranchInfoPtr = std::shared_ptr<ConditionSwitchBranchInfo>;

}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_HYPER_OFFLOAD_UTILS_H_
