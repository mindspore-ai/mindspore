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

#ifndef MINDSPORE_CORE_MINDRT_RUNTIME_CORE_AFFINITY_H_
#define MINDSPORE_CORE_MINDRT_RUNTIME_CORE_AFFINITY_H_

#include <vector>
#include "thread/threadpool.h"
#ifdef BIND_CORE

namespace mindspore {
class CoreAffinity {
 public:
  static CoreAffinity *GetInstance() {
    static CoreAffinity affinity;
    return &affinity;
  }
  int InitBindCoreId(size_t thread_num, BindMode bind_mode);

  int BindThreads(const std::vector<Worker *> &workers, const std::vector<int> &core_list);
  int BindThreads(const std::vector<Worker *> &workers, BindMode bind_mode) const;

 private:
  CoreAffinity() = default;
  ~CoreAffinity() = default;

  int BindThreadsToCoreList(const std::vector<Worker *> &workers) const;
  int FreeScheduleThreads(const std::vector<Worker *> &workers) const;
  int SetAffinity(pthread_t thread_id, cpu_set_t *cpuSet) const;
  int SortCPUProcessors();

  // bind_id contains the CPU cores to bind
  // the size of bind_id is equal to the size of workers
  std::vector<int> bind_id_;
  // sorted_id contains the ordered CPU core id
  // the size of sorted_id is equal to the size of hardware_concurrency
  std::vector<int> sorted_id_;
  size_t core_num_{0};
  size_t higher_num_{0};
  size_t thread_num_{0};
};
}  // namespace mindspore

#endif  // BIND_CORE
#endif  // MINDSPORE_CORE_MINDRT_RUNTIME_CORE_AFFINITY_H_
