/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef PREDICT_SRC_RUNTIME_THREAD_POOL_H_
#define PREDICT_SRC_RUNTIME_THREAD_POOL_H_

#include <mutex>
#include <condition_variable>
#include <thread>
#include <vector>
#include <string>
#include <atomic>
#include <memory>
#include <utility>
#include <functional>
#include "src/runtime/runtime_api.h"

namespace mindspore {
namespace predict {
constexpr int kSingleThreadMaxTask = 4;
using TvmEnv = TVMParallelGroupEnv;
using WorkFun = FTVMParallelLambda;
using TaskParam = struct Param {
  void *cdata;
  int32_t taskId;
  TvmEnv *tvmParam;
};
using ThreadPoolTask = std::pair<WorkFun, TaskParam>;

class LiteQueue {
 public:
  LiteQueue() = default;
  ~LiteQueue() = default;

  bool Enqueue(const ThreadPoolTask &task);
  bool Dequeue(ThreadPoolTask *out);
  std::atomic<int> taskSize{0};

 private:
  std::atomic<int> head{0};
  std::atomic<int> tail{0};
  ThreadPoolTask buffer[kSingleThreadMaxTask]{};
};

class LiteThreadBind {
 public:
  LiteThreadBind() = default;
  ~LiteThreadBind() = default;
  bool Bind(int numThreads, int mode);
  std::vector<std::pair<pthread_t, bool>> threadIdList;

 private:
  enum AffinityMode : int { BIG_CORE = 1, MID_CORE = -1, NO_BIND = 0 };
  void InitSortedCpuId();
  bool BindAllThread(bool bindFlag);
  bool BindMasterThread(bool bindFlag, int mode = MID_CORE);
  bool BindThreads(bool bindFlag);
  bool SetCPUBind(pthread_t threadId, const cpu_set_t &cpuSet);
  int bigCore{0};
  int midCore{0};
  int threadNums{0};
  std::vector<int> sortedCpuIds{};
  AffinityMode bindModel{MID_CORE};
};

class LiteThreadPool {
 public:
  LiteThreadPool() = default;
  explicit LiteThreadPool(int numThreads);
  ~LiteThreadPool();

  void AddNewThread(int newNums);
  bool DistributeTask(ThreadPoolTask task, int numTask);
  std::vector<std::thread> threadList{};

 private:
  using errCode = std::pair<bool, int>;
  bool AddRunReference();
  bool SubRunReference();
  bool CheckResult();
  int curThreadNums{0};
  std::vector<std::unique_ptr<LiteQueue>> queueList;
  std::atomic_int running{0};
  std::mutex tMutex;
  std::condition_variable queueReady;
  std::atomic<bool> destroy = {false};
  std::vector<std::pair<int, errCode>> errorInfo{};
};

class ThreadPool {
 public:
  static ThreadPool *GetInstance();
  void ConfigThreadPool(int mode, int numThreads);
  bool LaunchThreadPoolTask();
  bool AddTask(const WorkFun &worker, void *cdata, int numTask);

  ThreadPool(const ThreadPool &) = delete;
  ThreadPool &operator=(const ThreadPool &) = delete;

 private:
  ThreadPool() = default;
  ~ThreadPool() = default;
  int GetThreadNum(int numThreads);
  void GetThreadIdList();
  bool SetThreadPool(int numThreads = 1);
  bool SetThreadCpulBind(int mode);
  std::unique_ptr<LiteThreadPool> gThreadPool{nullptr};
  std::unique_ptr<LiteThreadBind> gThreadBind{nullptr};
  std::mutex gPoolMutex;
  int totalThreadNum{1};
  int bindMode{-1};
};
}  // namespace predict
}  // namespace mindspore

#endif  // PREDICT_SRC_RUNTIME_THREAD_POOL_H_
