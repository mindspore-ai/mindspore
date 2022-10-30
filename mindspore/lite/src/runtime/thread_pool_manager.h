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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_THREAD_POOL_MANAGER_H_
#define MINDSPORE_LITE_SRC_RUNTIME_THREAD_POOL_MANAGER_H_

#include <map>
#include <string>
#include <vector>
#include "thread/threadpool.h"

namespace mindspore {
namespace lite {
class ThreadPoolManager {
 public:
  static ThreadPoolManager *GetInstance() {
    static ThreadPoolManager instance;
    return &instance;
  }

  ~ThreadPoolManager();

  ThreadPool *GetThreadPool(size_t actor_num, size_t inter_op_parallel_num, size_t thread_num, BindMode bind_mode,
                            const std::vector<int> &core_list);

  void RetrieveThreadPool(size_t actor_num, size_t inter_op_parallel_num, size_t thread_num, BindMode bind_mode,
                          const std::vector<int> &core_list, ThreadPool *thread_pool);

 private:
  ThreadPoolManager() = default;

  std::string ComputeHash(size_t actor_num, size_t inter_op_parallel_num, size_t thread_num, BindMode bind_mode,
                          const std::vector<int> &core_list);

  std::map<std::string, std::vector<ThreadPool *>> thread_pool_container_;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_RUNTIME_THREAD_POOL_MANAGER_H_
