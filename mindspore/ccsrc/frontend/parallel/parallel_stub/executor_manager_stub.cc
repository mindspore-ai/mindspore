/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/parallel_stub/executor_manager_stub.h"
namespace mindspore {
namespace parallel {
std::shared_ptr<Executor> ExecutorManager::GetExecutor(const std::string &dev_name, uint32_t dev_id) {
  std::string dev_key = dev_name + "_" + std::to_string(dev_id);
  auto iter = executors_.find(dev_key);
  if (iter != executors_.end()) {
    return iter->second;
  }
  auto executor = std::make_shared<Executor>(dev_name, dev_id);
  executors_[dev_key] = executor;
  return executor;
}
}  // namespace parallel
}  // namespace mindspore
