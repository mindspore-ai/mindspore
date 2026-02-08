/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "include/utils/comm_manager.h"
#include "utils/ms_context.h"
#include "include/utils/parallel_context.h"

namespace mindspore {
namespace {
constexpr auto kDefaultCommManagerName = "default_comm_manager_name";
constexpr unsigned int kNoCommDlibRankSize = 2048;

std::map<std::string, std::shared_ptr<CommManager>> &GetInstanceMap() {
  static std::map<std::string, std::shared_ptr<CommManager>> kCommInstanceMap = {};
  return kCommInstanceMap;
}

class DefaultCommManager : public CommManager {
 public:
  DefaultCommManager() : CommManager("hccl") {}
  ~DefaultCommManager() override = default;

  bool CreateGroupSync(const string &, const std::vector<unsigned int> &) const override { return true; }
  bool GetRankID(const string &, unsigned int *) const override { return true; }
  bool GetRankSize(const string &, unsigned int *rank_size) const override {
    *rank_size = kNoCommDlibRankSize;
    return true;
  }

  bool DestroyGroup(const string &) const override { return true; }

  uint32_t GetRank() override {
    auto env_rank_id = common::GetEnv("RANK_ID");
    return env_rank_id.empty() ? 0 : std::stoi(env_rank_id);
  }
};
COMM_MANAGER_REG(kDefaultCommManagerName, std::make_shared<DefaultCommManager>());
}  // namespace

bool CommManager::Register(const std::string &name, const std::shared_ptr<CommManager> &instance) {
  if (GetInstanceMap().find(name) != GetInstanceMap().end()) {
    return false;
  }

  (void)GetInstanceMap().emplace(name, instance);
  return true;
}

void CommManager::Clear() { GetInstanceMap().clear(); }

CommManager &CommManager::GetInstance() noexcept {
  if (GetInstanceMap().empty()) {
    MS_LOG(EXCEPTION) << "No CommManager instance found.";
  }

  auto default_iter = GetInstanceMap().find(kDefaultCommManagerName);
  if (default_iter == GetInstanceMap().end()) {
    MS_LOG(EXCEPTION) << "Default CommManager instance not found.";
  }

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  std::string device_name = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (auto iter = GetInstanceMap().find(device_name); iter != GetInstanceMap().end()) {
    return *(iter->second);
  }

  if (static bool first_warning = true; first_warning) {
    MS_LOG(WARNING) << "CommManager instance for " << device_name << " not found, return default instance.";
    first_warning = false;
  }
  return *(default_iter->second);
}

uint32_t GetRank() { return CommManager::GetInstance().GetRank(); }

bool IsStandAlone() {
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  return parallel_context->parallel_mode() == parallel::kStandalone;
}
}  // namespace mindspore
