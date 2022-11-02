/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_SESSION_EXECUTOR_MANAGER_H_
#define MINDSPORE_CCSRC_BACKEND_SESSION_EXECUTOR_MANAGER_H_
#include <set>
#include <map>
#include <string>
#include <memory>
#include "backend/common/session/executor.h"
#include "include/backend/visible.h"

namespace mindspore::session {
class Executor;
class BACKEND_EXPORT ExecutorManager {
 public:
  static ExecutorManager &Instance();
  std::shared_ptr<Executor> GetExecutor(const std::string &device_name, uint32_t device_id);
  void OnEvent(const ExecutorEvent &event);
  void Clear();
  void ClearDoneTasks() {
    for (const auto &item : executors_) {
      auto &executor = item.second;
      if (executor != nullptr) {
        executor->ClearDoneTasks();
      }
    }
  }

 private:
  ExecutorManager() = default;
  ~ExecutorManager() = default;
  DISABLE_COPY_AND_ASSIGN(ExecutorManager)

  std::map<std::string, std::shared_ptr<Executor>> executors_;
};
}  // namespace mindspore::session
#endif  // MINDSPORE_CCSRC_BACKEND_SESSION_EXECUTOR_MANAGER_H_
