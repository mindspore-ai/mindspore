/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PS_CORE_SCHEDULER_RECOVERY_H_
#define MINDSPORE_CCSRC_PS_CORE_SCHEDULER_RECOVERY_H_

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "include/backend/distributed/ps/constants.h"
#include "utils/log_adapter.h"
#include "ps/core/file_configuration.h"
#include "include/backend/distributed/ps/ps_context.h"
#include "ps/core/recovery_base.h"
#include "ps/core/scheduler_node.h"
#include "ps/core/node_info.h"

namespace mindspore {
namespace ps {
namespace core {
// The class helps scheduler node to do recovery operation for the cluster.
class SchedulerRecovery : public RecoveryBase {
 public:
  SchedulerRecovery() = default;
  ~SchedulerRecovery() override = default;

  bool Recover() override;

  // Get metadata from storage.
  std::string GetMetadata(const std::string &key);

 private:
  // The node_ will only be instantiated with worker/server node.
  SchedulerNode *const node_ = nullptr;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_SCHEDULER_RECOVERY_H_
