/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include <thread>

#ifdef PARALLEL_INFERENCE
#define BIND_CORE
#endif
#ifdef __ANDROID__
#define BIND_CORE
#include <sched.h>
#endif
#ifdef _WIN32
#define BIND_CORE
#endif

namespace mindspore {
enum BindMode {
  Power_NoBind = 0,  // free schedule
  Power_Higher = 1,
  Power_Middle = 2,
};
#define PARSE_CPU_GAP 3
#define PARSE_CPU_DEC 10
#define PARSE_CPU_HEX 16

#ifdef _WIN32
void SetWindowsSelfAffinity(uint64_t core_id);
#endif

class Worker;
class CoreAffinity {
 public:
  CoreAffinity() = default;
  ~CoreAffinity() = default;

  int InitHardwareCoreInfo();

  int BindThreads(const std::vector<Worker *> &workers, const std::vector<int> &core_list);
  int BindThreads(const std::vector<Worker *> &workers, BindMode bind_mode);
  int BindProcess(BindMode bind_mode);
  std::vector<int> GetCoreId(size_t thread_num, BindMode bind_mode) const;
  void SetCoreId(const std::vector<int> &core_list);
  static float GetServerFrequency();

 private:
#ifdef _WIN32
  int SetAffinity();
#elif defined(BIND_CORE)
  int SetAffinity(const pthread_t &thread_id, cpu_set_t *cpu_set);
#endif

  int InitBindCoreId(size_t thread_num, BindMode bind_mode);

  int BindThreadsToCoreList(const std::vector<Worker *> &workers);
  int FreeScheduleThreads(const std::vector<Worker *> &workers);

  // bind_id contains the CPU cores to bind
  // the size of bind_id is equal to the size of workers
  std::vector<int> bind_id_;
  // sorted_id contains the ordered CPU core id
  // the size of sorted_id is equal to the size of hardware_concurrency
  std::vector<int> sorted_id_;
  // used to store the frequency of core
  // the core id corresponds to the index
  std::vector<int> core_freq_;
  size_t core_num_{0};
  size_t higher_num_{0};
};
}  // namespace mindspore

#endif  // MINDSPORE_CORE_MINDRT_RUNTIME_CORE_AFFINITY_H_
