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

#ifndef MINDSPORE_CORE_IR_PRIMAL_DEBUG_INFO_H
#define MINDSPORE_CORE_IR_PRIMAL_DEBUG_INFO_H
#include <string>
#include <memory>
#include <stack>
#include <vector>
#include <unordered_map>
#include "utils/info.h"

namespace mindspore {
class PrimalDebugInfoManager {
 public:
  static PrimalDebugInfoManager &GetInstance() noexcept {
    static PrimalDebugInfoManager instance;
    return instance;
  }
  PrimalDebugInfoManager(const PrimalDebugInfoManager &) = delete;
  PrimalDebugInfoManager &operator=(const PrimalDebugInfoManager &) = delete;
  ~PrimalDebugInfoManager() = default;
  void SetPrimalDebugInfo(const std::vector<NodeDebugInfoPtr> &primal_debug_infos) {
    primal_debug_infos_ = primal_debug_infos;
  }
  void ClearPrimalDebugInfo() { primal_debug_infos_.clear(); }
  std::vector<NodeDebugInfoPtr> GetCurrentPrimalDebugInfo() { return primal_debug_infos_; }

 private:
  PrimalDebugInfoManager() = default;
  std::vector<NodeDebugInfoPtr> primal_debug_infos_;
};

// PrimalDebugInfoGuard is a class that help generate the back propagation cnode
// with specified primal_debug_infos in the current c++ action scope.
class PrimalDebugInfoGuard {
 public:
  explicit PrimalDebugInfoGuard(const std::vector<NodeDebugInfoPtr> &primal_debug_infos) {
    PrimalDebugInfoManager::GetInstance().SetPrimalDebugInfo(primal_debug_infos);
  }
  ~PrimalDebugInfoGuard() { PrimalDebugInfoManager::GetInstance().ClearPrimalDebugInfo(); }
};
}  // namespace mindspore
#endif  // MINDSPORE_CORE_IR_PRIMAL_DEBUG_INFO_H
