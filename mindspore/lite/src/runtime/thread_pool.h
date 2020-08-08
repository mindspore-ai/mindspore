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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_THREAD_POOL_H_
#define MINDSPORE_LITE_SRC_RUNTIME_THREAD_POOL_H_

#include <mutex>
#include <condition_variable>
#include <thread>
#include <vector>
#include <string>
#include <atomic>
#include <memory>
#include <utility>
#include <functional>
#include <iostream>
#include "src/runtime/runtime_api.h"

namespace mindspore {
namespace predict {
#ifndef CPU_SET
const int CPU_SETSIZE = 1024;
#define __NCPUBITS (8 * sizeof(uint64_t))
typedef struct {
  uint64_t __bits[CPU_SETSIZE / __NCPUBITS];
} cpu_set_t;

#define CPU_SET_LOCAL(cpu, cpusetp) ((cpusetp)->__bits[(cpu) / __NCPUBITS] |= (1UL << ((cpu) % __NCPUBITS)))
#endif

constexpr int kSingleThreadMaxTask = 2;
using TvmEnv = LiteParallelGroupEnv;
using WorkFun = std::function<int(int, TvmEnv *, void *)>;
using TaskParam = struct Param {
  void *cdata;
  TvmEnv *tvmParam;
};
using ThreadPoolTask = std::pair<WorkFun, TaskParam>;
enum AffinityMode : int { BIG_CORE = 1, MID_CORE = -1, NO_BIND = 0 };

class LiteQueue {
 public:
  LiteQueue() = default;
  ~LiteQueue() = default;
  bool Enqueue(ThreadPoolTask *task);
  bool Dequeue(ThreadPoolTask **out);
  std::atomic_int taskSize = {0};

 private:
  std::atomic_int head = {0};
  std::atomic_int tail = {0};
  ThreadPoolTask *buffer[kSingleThreadMaxTask]{};
};

class LiteThreadBind {
 public:
  LiteThreadBind() = default;
  ~LiteThreadBind() = default;
  void InitSortedCpuId();
  bool Bind(bool ifBind, int numThreads, bool master);
  AffinityMode bindModel = MID_CORE;
  std::vector<pthread_t> threadIdList;

 private:
  bool BindMasterThread(bool bindFlag, int mode);
  bool BindThreads(bool bindFlag);
  bool SetCPUBind(pthread_t threadId, cpu_set_t *cpuSet);
  int bigCore = 0;
  int midCore = 0;
  std::vector<unsigned int> sortedCpuIds{};
};

class ThreadPool {
 public:
  ThreadPool() = default;
  ~ThreadPool();
  static ThreadPool *GetInstance();
  bool LaunchWork(WorkFun worker, void *cdata, int numTask);
  void ConfigThreadPool(int mode, int numThreads);
  void ConfigMaxThreadNum(unsigned int num);
  bool BindAllThreads(bool ifBind, int mode, bool master = true);
  ThreadPool(const ThreadPool &) = delete;
  ThreadPool &operator=(const ThreadPool &) = delete;

 private:
  bool SetThreadPool();
  void AddNewThread(int newNums);
  bool SetThreadCpuBind(bool ifBind, int mode, bool master);
  bool AddTask(WorkFun &&worker, void *cdata, int numTask);
  bool DistributeTask(ThreadPoolTask *task, int numTask);
  void AddRunThread(int num);
  void SubRunThread(int num);
  bool CheckResult();

  std::mutex poolMutex;
  std::mutex tMutex;
  std::condition_variable queueReady;
  std::atomic_bool exitRun = {false};
  std::vector<std::atomic_bool *> activateList{};
  int curThreadNums = 1;
  int curThreadRunNums = 1;
  int configThreadNums = 1;
  int configBindMode = -1;
  std::vector<std::thread> threadList{};
  std::vector<std::shared_ptr<LiteQueue>> queueList{};
  std::unique_ptr<LiteThreadBind> threadBind{nullptr};
  std::vector<std::pair<int, std::pair<bool, int>>> errorInfo{};
};
}  // namespace predict
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_RUNTIME_THREAD_POOL_H_

