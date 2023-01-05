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

#include "src/litert/thread_pool_reuse_manager.h"
#include <mutex>
#include "src/common/log_adapter.h"

namespace mindspore {
namespace lite {
namespace {
std::mutex l;
}  // namespace

ThreadPoolReuseManager::~ThreadPoolReuseManager() {
  std::lock_guard<std::mutex> lock(l);
  for (auto &pair : thread_pool_container_) {
    for (auto &thread_pool : pair.second) {
      if (thread_pool) {
        delete thread_pool;
      }
      thread_pool = nullptr;
    }
  }
  thread_pool_container_.clear();
}

ThreadPool *ThreadPoolReuseManager::GetThreadPool(size_t actor_num, size_t inter_op_parallel_num, size_t thread_num,
                                                  BindMode bind_mode, const std::vector<int> &core_list,
                                                  std::string runner_id) {
#ifdef SERVER_INFERENCE
  auto hash_key = ComputeHash(actor_num, inter_op_parallel_num, thread_num, bind_mode, core_list);
  std::lock_guard<std::mutex> lock(l);
  if (thread_pool_container_.find(hash_key) == thread_pool_container_.end()) {
    return nullptr;
  }
  if (thread_pool_container_[hash_key].empty()) {
    return nullptr;
  }
  auto thread_pool = thread_pool_container_[hash_key].back();
  if (inter_op_parallel_num > 1 && !thread_pool->SetRunnerID(runner_id)) {
    MS_LOG(WARNING) << "can not reuse thread pool.";
    return nullptr;
  }
  thread_pool_container_[hash_key].pop_back();
  return thread_pool;
#else
  return nullptr;
#endif
}

void ThreadPoolReuseManager::RetrieveThreadPool(size_t actor_num, size_t inter_op_parallel_num, size_t thread_num,
                                                BindMode bind_mode, const std::vector<int> &core_list,
                                                ThreadPool *thread_pool) {
  if (thread_pool == nullptr) {
    return;
  }
#ifdef SERVER_INFERENCE
  auto hash_key = ComputeHash(actor_num, inter_op_parallel_num, thread_num, bind_mode, core_list);
  std::lock_guard<std::mutex> lock(l);
  thread_pool_container_[hash_key].push_back(thread_pool);
#else
  delete thread_pool;
#endif
}

std::string ThreadPoolReuseManager::ComputeHash(size_t actor_num, size_t inter_op_parallel_num, size_t thread_num,
                                                BindMode bind_mode, const std::vector<int> &core_list) {
  std::string hash_key = std::to_string(actor_num) + std::to_string(inter_op_parallel_num) +
                         std::to_string(thread_num) + std::to_string(bind_mode);
  for (auto val : core_list) {
    hash_key += std::to_string(val);
  }
  return hash_key;
}
}  // namespace lite
}  // namespace mindspore
