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

#include "src/runtime/thread_pool.h"
#include <algorithm>
#include "utils/log_adapter.h"
#ifdef MS_COMPILE_IOS
#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/machine.h>
#endif  // MS_COMPILE_IOS

namespace mindspore {
namespace predict {
constexpr int kDefaultBigCount = 2;
constexpr int kDefaultMidCount = 2;
constexpr int kSmallCpuNum = 4;
constexpr int kBigMidCpuNum = 4;
constexpr int kDefaultThreadNum = 1;
static unsigned int kDefaultMaxThreadNums = 8;
static unsigned int localMaxThreadNums = 1;

bool LiteQueue::Enqueue(ThreadPoolTask *task) {
  const int tailIndex = tail.load(std::memory_order_relaxed);
  // queue full
  auto next = (tailIndex + 1) % kSingleThreadMaxTask;
  if (next == head.load(std::memory_order_acquire)) {
    return false;
  }
  buffer[tailIndex] = task;
  tail.store(next, std::memory_order_release);
  ++taskSize;
  return true;
}

bool LiteQueue::Dequeue(ThreadPoolTask **out) {
  if (taskSize == 0) {
    return false;
  }
  // queue empty
  const int headIndex = head.load(std::memory_order_relaxed);
  if (headIndex == tail.load(std::memory_order_acquire)) {
    return false;
  }
  *out = buffer[headIndex];
  head.store((headIndex + 1) % kSingleThreadMaxTask, std::memory_order_release);
  return true;
}

bool LiteThreadBind::Bind(bool ifBind, int numThreads, bool master) {
  if (master) {
    if (!BindMasterThread(ifBind, bindModel)) {
      MS_LOG(ERROR) << "bind msater thread failed";
      return false;
    }
    MS_LOG(DEBUG) << "bind master thread successful";
  }
  if (numThreads > static_cast<int>(sortedCpuIds.size())) {
    MS_LOG(ERROR) << "thread num " << numThreads << " is larger than cores " << static_cast<int>(sortedCpuIds.size())
                  << " in the system";
    return true;
  }

  if (!BindThreads(ifBind)) {
    MS_LOG(ERROR) << "action " << ifBind << " thread failed";
    return false;
  }
  MS_LOG(DEBUG) << "action " << ifBind << " thread successful";
  return true;
}

void LiteThreadBind::InitSortedCpuId() {
  // mate10(970)|p20(970): 4big, 4small
  // mate20(980)|p30(980)|mate30(990): 2big, 2mid, 4small
  // note: p30's core 7 not allowed to be bind
  int numCores = 0;
#ifdef MS_COMPILE_IOS
  size_t len = sizeof(numCores);
  sysctlbyname("hw.ncpu", &numCores, &len, NULL, 0);
  numCores = numCores > 1 ? numCores : 1;
#else
  numCores = static_cast<int>(std::thread::hardware_concurrency());
#endif  // MS_COMPILE_IOS
  if (numCores < kBigMidCpuNum) {
    bigCore = 0;
    midCore = numCores;
  } else {
    bigCore = kDefaultBigCount;
    midCore = kDefaultMidCount;
  }
  sortedCpuIds.clear();
  for (int i = numCores - 1; i >= 0; --i) {
    sortedCpuIds.emplace_back(i);
  }
  if (sortedCpuIds.size() > kSmallCpuNum) {
    sortedCpuIds.resize(bigCore + midCore);
  }
}

bool LiteThreadBind::BindMasterThread(bool bindFlag, int mode) {
  std::vector<int> cpu;
  if (bindFlag) {
    size_t cpuIndex;
    if (mode == MID_CORE) {
      cpuIndex = sortedCpuIds.size() - 1;
    } else {
      cpuIndex = 0;
    }
    cpu.emplace_back(sortedCpuIds[cpuIndex]);
  } else {
    // unbind master
    cpu.assign(sortedCpuIds.begin(), sortedCpuIds.end());
  }
  cpu_set_t cpuSet;
#ifndef CPU_SET
  (void)memset(&cpuSet, 0, sizeof(cpu_set_t));
#else
  CPU_ZERO(&cpuSet);
#endif
  for (auto coreId : cpu) {
#ifndef CPU_SET
    CPU_SET_LOCAL(coreId, &cpuSet);
#else
    CPU_SET(coreId, &cpuSet);
#endif
  }
  if (!SetCPUBind(pthread_self(), &cpuSet)) {
    MS_LOG(ERROR) << "do master bind failed. mode: " << mode;
    return false;
  }
  return true;
}

bool LiteThreadBind::BindThreads(bool bindFlag) {
  if (bindFlag && bindModel != NO_BIND) {
    size_t bindNums = std::min(sortedCpuIds.size(), threadIdList.size());
    cpu_set_t cpuSet;
    size_t coreIndex;
    for (size_t i = 0; i < bindNums; ++i) {
#ifndef CPU_SET
      (void)memset(&cpuSet, 0, sizeof(cpu_set_t));
#else
      CPU_ZERO(&cpuSet);
#endif
      if (bindModel == MID_CORE) {
        coreIndex = sortedCpuIds.size() - 2 - i;
      } else {
        coreIndex = i + 1;
      }
#ifndef CPU_SET
      CPU_SET_LOCAL(sortedCpuIds[coreIndex], &cpuSet);
#else
      CPU_SET(sortedCpuIds[coreIndex], &cpuSet);
#endif
      if (!SetCPUBind(threadIdList[i], &cpuSet)) {
        MS_LOG(ERROR) << "do SetCPUBind failed";
        return false;
      }
    }
  } else {
    // unbind
    size_t bindNums = std::min(sortedCpuIds.size(), threadIdList.size());
    cpu_set_t cpuSet;
#ifndef CPU_SET
    (void)memset(&cpuSet, 0, sizeof(cpu_set_t));
#else
    CPU_ZERO(&cpuSet);
#endif
    for (auto coreId : sortedCpuIds) {
#ifndef CPU_SET
      CPU_SET_LOCAL(coreId, &cpuSet);
#else
      CPU_SET(coreId, &cpuSet);
#endif
    }
    for (size_t i = 0; i < bindNums; ++i) {
      if (!SetCPUBind(threadIdList[i], &cpuSet)) {
        MS_LOG(ERROR) << "do SetCPUBind failed";
        return false;
      }
    }
  }
  return true;
}

bool LiteThreadBind::SetCPUBind(pthread_t threadId, cpu_set_t *cpuSet) {
#if defined(__ANDROID__)
#if __ANDROID_API__ >= 21
  int ret = sched_setaffinity(pthread_gettid_np(threadId), sizeof(cpu_set_t), cpuSet);
  if (ret != 0) {
    MS_LOG(ERROR) << "bind thread " << threadId << "to cpu failed.ERROR " << ret;
  }
#endif
#else
#ifdef __APPLE__
  MS_LOG(ERROR) << "not bind thread to apple's cpu.";
  return false;
#else
  int ret = pthread_setaffinity_np(threadId, sizeof(cpuSet), cpuSet);
  if (ret != 0) {
    MS_LOG(ERROR) << "bind thread " << threadId << " to cpu failed.ERROR " << ret;
    return false;
  }
#endif  // __APPLE__
#endif
  return true;
}

bool ThreadPool::SetThreadPool() {
  std::lock_guard<std::mutex> Lock(poolMutex);
  if (configThreadNums <= 0) {
    MS_LOG(WARNING) << "numThreads " << configThreadNums << ", must be greater than 0";
    configThreadNums = curThreadRunNums;
  }
  if (localMaxThreadNums == 0) {
    localMaxThreadNums = 1;
  } else if (localMaxThreadNums > kDefaultMaxThreadNums) {
    localMaxThreadNums = kDefaultMaxThreadNums;
  }
  if (configThreadNums > kDefaultMaxThreadNums) {
    configThreadNums = kDefaultMaxThreadNums;
  }
  int addNum = 0;
  if (configThreadNums > kDefaultMaxThreadNums) {
    addNum = configThreadNums - curThreadRunNums;
  } else if (localMaxThreadNums > curThreadNums) {
    addNum = localMaxThreadNums - curThreadNums;
  }
  AddNewThread(addNum);
  if (curThreadRunNums > localMaxThreadNums) {
    SubRunThread(localMaxThreadNums);
  } else {
    AddRunThread(localMaxThreadNums);
  }
  return true;
}

void ThreadPool::AddNewThread(int newNums) {
  for (int i = curThreadNums - 1, j = 0; j < newNums; ++i, ++j) {
    auto active = new std::atomic_bool{true};
    auto queue = std::make_shared<LiteQueue>();
    threadList.emplace_back([this, i, active, queue]() {
      ThreadPoolTask *task = nullptr;
      while (!exitRun) {
        while (*active) {
          if (queue->Dequeue(&task)) {
            auto ret = task->first(i + 1, task->second.tvmParam, task->second.cdata);
            if (ret != 0) {
              errorInfo.emplace_back(std::make_pair(i + 1, std::make_pair(false, ret)));
            }
            queue->taskSize--;
          }
          std::this_thread::yield();
        }
        std::unique_lock<std::mutex> queueLock(tMutex);
        queueReady.wait(queueLock, [active, this] { return exitRun || *active; });
      }
    });
    activateList.emplace_back(active);
    queueList.emplace_back(queue);
  }
  curThreadNums += newNums;
  curThreadRunNums += newNums;
}

bool ThreadPool::SetThreadCpuBind(bool ifBind, int mode, bool master) {
  if (curThreadRunNums <= 0) {
    MS_LOG(ERROR) << "no threads need to be bind, totalThreadNum : " << curThreadRunNums;
    return false;
  }
  if (threadBind == nullptr) {
    threadBind = std::unique_ptr<LiteThreadBind>(new LiteThreadBind());
    if (threadBind == nullptr) {
      MS_LOG(ERROR) << "create threadBind failed";
      return false;
    }
    threadBind->threadIdList.resize(kDefaultMaxThreadNums);
    threadBind->InitSortedCpuId();
  }
  threadBind->threadIdList.clear();
  for (auto &it : threadList) {
    threadBind->threadIdList.emplace_back(it.native_handle());
  }
  threadBind->bindModel = static_cast<AffinityMode>(mode);
  if (!threadBind->Bind(ifBind, curThreadRunNums, master)) {
    MS_LOG(ERROR) << "bind failed";
    return false;
  }
  return true;
}

bool ThreadPool::AddTask(WorkFun &&worker, void *cdata, int numTask) {
  if (numTask <= 0) {
    numTask = curThreadRunNums;
  }
  TvmEnv env{};
  env.num_task = numTask;
  errorInfo.clear();
  // single task, run master thread
  if (curThreadRunNums <= 1) {
    for (int i = 0; i < numTask; ++i) {
      int ret = worker(i, &env, cdata);
      if (ret != 0) {
        errorInfo.emplace_back(std::make_pair(0, std::make_pair(false, ret)));
      }
    }
    return CheckResult();
  }
  ThreadPoolTask task;
  task.first = std::move(worker);
  task.second.cdata = cdata;
  task.second.tvmParam = &env;
  return DistributeTask(&task, numTask);
}

bool ThreadPool::DistributeTask(ThreadPoolTask *task, int numTask) {
  auto taskOri = *task;
  if (numTask > curThreadRunNums) {
    task->first = [taskOri, numTask, this](int task_id, TvmEnv *penv, void *cdata) -> int {
      for (int i = task_id; i < numTask; i += curThreadRunNums) {
        int ret = taskOri.first(i, penv, cdata);
        if (ret != 0) {
          errorInfo.emplace_back(std::make_pair(i + 1, std::make_pair(false, ret)));
        }
      }
      return 0;
    };
  }
  bool kSuccFlag;
  auto size = std::min(curThreadRunNums, numTask);
  for (int i = 0; i < size - 1; ++i) {
    do {
      kSuccFlag = true;
      if (!queueList[i]->Enqueue(task)) {
        std::this_thread::yield();
        kSuccFlag = false;
      }
    } while (!kSuccFlag);
  }
  // master thread
  int ret = task->first(0, task->second.tvmParam, task->second.cdata);
  if (ret != 0) {
    errorInfo.emplace_back(std::make_pair(0, std::make_pair(false, ret)));
  }
  kSuccFlag = false;
  while (!kSuccFlag) {
    std::this_thread::yield();
    kSuccFlag = true;
    for (int i = 0; i < curThreadRunNums - 1; ++i) {
      if (queueList[i]->taskSize != 0) {
        kSuccFlag = false;
        break;
      }
    }
  }
  return CheckResult();
}

void ThreadPool::AddRunThread(int num) {
  int activeNums = num - curThreadRunNums;
  if (activeNums <= 0 || activateList.size() < activeNums) {
    return;
  }
  for (int i = curThreadRunNums - 1, j = 0; j < activeNums; ++i, ++j) {
    *activateList[i] = true;
  }
  std::lock_guard<std::mutex> queueLock(tMutex);
  queueReady.notify_all();
  curThreadRunNums = num;
}

void ThreadPool::SubRunThread(int num) {
  int deactiveNums = curThreadRunNums - num;
  if (deactiveNums <= 0) {
    return;
  }
  for (int i = num - 1, j = 0; j < deactiveNums; ++i, ++j) {
    *activateList[i] = false;
  }
  curThreadRunNums = num;
}

bool ThreadPool::CheckResult() {
  bool kSuccFlag = true;
  for (auto result : errorInfo) {
    if (result.second.first) {
      MS_LOG(ERROR) << "task " << result.first << " failed, error code is " << result.second.second;
      kSuccFlag = false;
    }
  }
  return kSuccFlag;
}

bool ThreadPool::LaunchWork(WorkFun worker, void *cdata, int numTask) {
  if (!SetThreadPool()) {
    return false;
  }
  return AddTask(std::move(worker), cdata, numTask);
}

bool ThreadPool::BindAllThreads(bool ifBind, int mode, bool master) {
  if (!SetThreadPool()) {
    return false;
  }
  return SetThreadCpuBind(ifBind, mode, master);
}

void ThreadPool::ConfigThreadPool(int mode, int numThreads) {
  configBindMode = mode;
  configThreadNums = numThreads;
}

void ThreadPool::ConfigMaxThreadNum(unsigned int num) { localMaxThreadNums = num; }

ThreadPool *ThreadPool::GetInstance() {
  static ThreadPool instance;
  return &instance;
}

ThreadPool::~ThreadPool() {
  curThreadRunNums = static_cast<int>(threadList.size() + 1);
  exitRun = true;
  SubRunThread(kDefaultThreadNum);
  queueReady.notify_all();
  for (auto &it : threadList) {
    if (it.joinable()) {
      it.join();
    }
  }
  for (const auto &it : activateList) {
    delete it;
  }
}
}  // namespace predict
}  // namespace mindspore

