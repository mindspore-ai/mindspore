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
#ifndef MINDSPORE_CCSRC_PARALLEL_EXECUTOR_MANAGER_STUB_H_
#define MINDSPORE_CCSRC_PARALLEL_EXECUTOR_MANAGER_STUB_H_
#include <set>
#include <map>
#include <string>
#include <memory>
#include "frontend/parallel/parallel_stub/executor_stub.h"
namespace mindspore {
namespace parallel {
class Executor;
class ExecutorManager {
 public:
  static ExecutorManager &Instance() {
    static ExecutorManager instance;
    return instance;
  }
  std::shared_ptr<Executor> GetExecutor(const std::string &device_name, int device_id);

 private:
  ExecutorManager() = default;
  ~ExecutorManager() = default;
  std::map<std::string, std::shared_ptr<Executor>> executors_;
};
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PARALLEL_EXECUTOR_MANAGER_STUB_H_
