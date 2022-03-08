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

#ifndef MINDSPORE_CCSRC_PS_CORE_PS_SCHEDULER_NODE_H_
#define MINDSPORE_CCSRC_PS_CORE_PS_SCHEDULER_NODE_H_

#include "ps/core/scheduler_node.h"
#include "ps/core/node_info.h"

namespace mindspore {
namespace ps {
namespace core {
// This class is a derived class of SchedulerNode specialized for Parameter Server. It is used to rewrite the specific
// logic for Parameter Server mode training in SchedulerNode. For example, the Scheduler of Parameter Server will reject
// the registration request of alive nodes.
class PSSchedulerNode : public SchedulerNode {
 public:
  PSSchedulerNode() = default;
  ~PSSchedulerNode() override = default;

 protected:
  // Override the scheduler node to remove the nofification from scheduler to other nodes.
  void RunRecovery() override;

 private:
  // Determine whether the registration request of the node should be rejected, the registration of the
  // alive node should be rejected.
  bool NeedRejectRegister(const NodeInfo &node_info) override { return node_info.is_alive; }
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PS_CORE_PS_SCHEDULER_NODE_H_
