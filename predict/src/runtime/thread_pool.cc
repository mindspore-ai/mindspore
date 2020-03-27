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

#include "src/runtime/thread_pool.h"
#include <algorithm>
#include "common/mslog.h"

namespace mindspore {
namespace predict {
static constexpr int kThreadPoolMaxThreads = 8;
static const int kCoreNumThr = 4;
static const int kMidCoreNum = 2;
static const int kBigCoreNum = 2;
bool LiteQueue::Enqueue(const ThreadPoolTask &task) {
  const int tailIndex = tail.load(std::memory_order_relaxed);
  // queue full
  auto next = (tailIndex + 1) % kSingleThreadMaxTask;
  if (next == head.load(std::memory_order_acquire)) {
    return false;
  }
  buffer[tailIndex] = task;
  tail.store(next, std::memory_order_release);
  taskSize.fetch_add(1);
  return true;
}

bool LiteQueue::Dequeue(ThreadPoolTask *out) {
  if (out == nullptr) {
    MS_LOGE("ThreadPoolTask is nullptr");
    return false;
  }
  if (taskSize.load() == 0) {
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

bool LiteThreadBind::Bind(int numThreads, int mode) {
  InitSortedCpuId();
  if (numThreads > static_cast<int>(sortedCpuIds.size())) {
    MS_LOGE("thread num %d is larger than cores %lu in the system", numThreads, sortedCpuIds.size());
    return false;
  }
  threadNums = numThreads + 1;
  bindModel = static_cast<AffinityMode>(mode);
  if (bindModel == NO_BIND) {
    if (!BindAllThread(false)) {
      MS_LOGE("unbind %d threads failed", threadNums);
      return false;
    }
    MS_LOGD("unbind %d threads successful", threadNums);
  } else {
    if (!BindAllThread(true)) {
      MS_LOGE("bind %d threads failed", threadNums);
      return false;
    }
    MS_LOGD("bind %d threads successful", threadNums);
  }
  return true;
}

void LiteThreadBind::InitSortedCpuId() {
  int numCores = static_cast<int>(std::thread::hardware_concurrency());
  if (numCores < kCoreNumThr) {
    bigCore = 0;
    midCore = numCores;
  } else {
    bigCore = kBigCoreNum;
    midCore = kMidCoreNum;
  }
  if (numCores > kCoreNumThr) {
    numCores = bigCore + midCore;
  }
  sortedCpuIds.resize(numCores);
  sortedCpuIds.clear();
  for (int i = numCores - 1; i >= 0; --i) {
    sortedCpuIds.emplace_back(i);
  }
}

bool LiteThreadBind::BindAllThread(bool bindFlag) {
  if (threadNums <= 0) {
    MS_LOGE("no thread pool find, current threadNums %d", threadNums);
    return false;
  }
  if (!BindThreads(bindFlag)) {
    MS_LOGE("bind threads failed");
    return false;
  }
  return true;
}

bool LiteThreadBind::BindMasterThread(bool bindFlag, int mode) {
  std::vector<int> cpu;
  cpu.resize(sortedCpuIds.size());
  cpu.clear();
  if (bindFlag) {
    int cpuIndex = (mode == MID_CORE) ? (threadNums - 1) : 0;
    auto materCpuId = sortedCpuIds.at(cpuIndex);
    cpu.emplace_back(materCpuId);
  } else {
    // unbind master
    cpu.assign(sortedCpuIds.begin(), sortedCpuIds.end());
  }
  cpu_set_t cpuSet;
  CPU_ZERO(&cpuSet);
  for (auto coreId : cpu) {
    CPU_SET(coreId, &cpuSet);
  }
  if (!SetCPUBind(pthread_self(), cpuSet)) {
    MS_LOGE("do master bind failed. mode: %d", mode);
    return false;
  }
  return true;
}

bool LiteThreadBind::BindThreads(bool bindFlag) {
  if (bindFlag) {
    if (bindModel != NO_BIND) {
      size_t bindNums = std::min(sortedCpuIds.size(), threadIdList.size());
      size_t coreIndex;
      cpu_set_t cpuSet;
      for (size_t i = 0; i < bindNums; ++i) {
        if (bindModel == MID_CORE) {
          coreIndex = sortedCpuIds.size() - i - 1;
        } else {
          coreIndex = i;
        }
        CPU_ZERO(&cpuSet);
        CPU_SET(sortedCpuIds[coreIndex], &cpuSet);
        if (!threadIdList[i].second) {
          MS_LOGD("threadIdList[%lu]=%lu, sortedCpuIds[%lu]=%d", i, threadIdList[i].first, coreIndex,
                  sortedCpuIds[coreIndex]);
          if (!SetCPUBind(threadIdList[i].first, cpuSet)) {
            MS_LOGE("do SetCPUBind failed");
            return false;
          }
        }
        threadIdList[i].second = true;
      }
    }
  } else {
    // unbind
    size_t bindNums = std::min(sortedCpuIds.size(), threadIdList.size());
    cpu_set_t cpuSet;
    CPU_ZERO(&cpuSet);
    for (auto coreId : sortedCpuIds) {
      CPU_SET(coreId, &cpuSet);
    }
    for (size_t i = 0; i < bindNums; ++i) {
      if (!SetCPUBind(threadIdList[i].first, cpuSet)) {
        MS_LOGE("do SetCPUBind failed");
        return false;
      }
      threadIdList[i].second = false;
    }
  }
  return true;
}

bool LiteThreadBind::SetCPUBind(pthread_t threadId, const cpu_set_t &cpuSet) {
#if defined(__ANDROID__)
#if __ANDROID_API__ >= 21
  int ret = sched_setaffinity(pthread_gettid_np(threadId), sizeof(cpu_set_t), &cpuSet);
  if (ret != 0) {
    MS_LOGE("bind thread %ld to cpu failed.ERROR %d", threadId, ret);
  }
#endif
#else
  int ret = pthread_setaffinity_np(threadId, sizeof(cpu_set_t), &cpuSet);
  if (ret != 0) {
    MS_LOGE("bind thread %ld to cpu failed.ERROR %d", threadId, ret);
    return false;
  }
#endif
  return true;
}

LiteThreadPool::LiteThreadPool(int numThreads) {
  queueList.resize(kThreadPoolMaxThreads);
  queueList.clear();
  AddNewThread(numThreads);
}

void LiteThreadPool::AddNewThread(int newNums) {
  for (int i = curThreadNums, j = 0; j < newNums; ++j, ++i) {
    queueList.push_back(std::unique_ptr<LiteQueue>(new LiteQueue()));
    threadList.emplace_back([this, i]() {
      ThreadPoolTask task;
      while (!destroy) {
        while (running != 0) {
          MS_LOGD("i = %d, thread id = %lu, taskSize = %d", i, pthread_self(), queueList[i]->taskSize.load());
          while (queueList[i]->taskSize.load() > 0 && queueList[i]->Dequeue(&task)) {
            auto ret = task.first(task.second.taskId, task.second.tvmParam, task.second.cdata);
            if (ret != 0) {
              errorInfo.emplace_back(std::make_pair(task.second.taskId, std::make_pair(false, ret)));
            }
            queueList[i]->taskSize.fetch_sub(1);
          }
          std::this_thread::yield();
        }
        std::unique_lock<std::mutex> queueLock(tMutex);
        queueReady.wait(queueLock, [this] { return destroy || running != 0; });
      }
    });
  }
  MS_LOGI("%d new thread create", newNums);
  curThreadNums += newNums;
}

bool LiteThreadPool::DistributeTask(ThreadPoolTask task, int numTask) {
  // wake up
  errorInfo.clear();
  if (!AddRunReference()) {
    MS_LOGE("add reference failed");
    return false;
  }
  bool kSuccFlag;
  for (int i = 1; i < numTask; ++i) {
    task.second.taskId = i;
    do {
      kSuccFlag = false;
      for (auto &queue : queueList) {
        MS_ASSERT(queue != nullptr);
        if (queue->Enqueue(task)) {
          kSuccFlag = true;
          break;
        }
      }
      std::this_thread::yield();
    } while (!kSuccFlag);
  }
  MS_LOGI("add %d task successful", numTask);
  // master thread
  int ret = task.first(0, task.second.tvmParam, task.second.cdata);
  if (ret != 0) {
    errorInfo.emplace_back(std::make_pair(0, std::make_pair(false, ret)));
  }
  kSuccFlag = false;
  while (!kSuccFlag) {
    kSuccFlag = true;
    for (auto iter = queueList.begin(); iter != queueList.end(); ++iter) {
      if ((*iter)->taskSize.load() != 0) {
        kSuccFlag = false;
        break;
      }
    }
    std::this_thread::yield();
  }
  // hibernate
  if (!SubRunReference()) {
    MS_LOGE("sub reference failed");
    return false;
  }
  MS_LOGI("finish %d task successful", numTask);
  return CheckResult();
}

bool LiteThreadPool::AddRunReference() {
  running.fetch_add(1);
  std::lock_guard<std::mutex> queueLock(tMutex);
  queueReady.notify_all();
  return true;
}

bool LiteThreadPool::SubRunReference() {
  running.fetch_sub(1);
  return true;
}

bool LiteThreadPool::CheckResult() {
  bool kSuccFlag = true;
  for (auto result : errorInfo) {
    if (result.second.first) {
      MS_LOGE("task %d failed, error code is %d", result.first, result.second.second);
      kSuccFlag = false;
    }
  }
  return kSuccFlag;
}

int ThreadPool::GetThreadNum(int numThreads) {
  if (numThreads <= 0 || numThreads > kThreadPoolMaxThreads) {
    MS_LOGE("numThreads %d, must be greater than 0 or less than or equal to %d", numThreads, kThreadPoolMaxThreads);
    return -1;
  } else {
    if (numThreads > totalThreadNum) {
      return (numThreads - totalThreadNum);
    } else {
      MS_LOGD("%d threads have been already created", numThreads);
      return 0;
    }
  }
}

void ThreadPool::GetThreadIdList() {
  if (gThreadPool != nullptr) {
    for (int i = 0; i < totalThreadNum; ++i) {
      bool kSuccFlag = false;
      pthread_t threadHandle;
      do {
        kSuccFlag = false;
        threadHandle = gThreadPool->threadList[i].native_handle();
        if (threadHandle != 0) {
          kSuccFlag = true;
        }
        std::this_thread::yield();
      } while (!kSuccFlag);

      auto iter = std::find_if(std::begin(gThreadBind->threadIdList), std::end(gThreadBind->threadIdList),
                               [threadHandle](std::pair<pthread_t, bool> id) { return id.first == threadHandle; });
      if (iter == std::end(gThreadBind->threadIdList)) {
        gThreadBind->threadIdList.emplace_back(std::make_pair(threadHandle, false));
      }
    }
  }
  MS_ASSERT(gThreadBind != nullptr);
  gThreadBind->threadIdList.emplace_back(std::make_pair(pthread_self(), false));
}

bool ThreadPool::SetThreadCpulBind(int mode) {
  if (totalThreadNum <= 0) {
    MS_LOGE("no threads need to be bind, totalThreadNum : %d", totalThreadNum);
    return false;
  }
  std::lock_guard<std::mutex> bMutex(gPoolMutex);
  if (gThreadBind == nullptr) {
    gThreadBind = std::unique_ptr<LiteThreadBind>(new (std::nothrow) LiteThreadBind());
    if (gThreadBind == nullptr) {
      MS_LOGE("new LiteThreadBind failed");
      return false;
    }
    gThreadBind->threadIdList.resize(kThreadPoolMaxThreads + 1);
    gThreadBind->threadIdList.clear();
  }
  GetThreadIdList();

  if (!gThreadBind->Bind(totalThreadNum, mode)) {
    MS_LOGE("BindCore failed");
    return false;
  }
  return true;
}

bool ThreadPool::SetThreadPool(int numThreads) {
  std::lock_guard<std::mutex> Lock(gPoolMutex);
  int realNums = GetThreadNum(numThreads);
  if (realNums < -1) {
    return false;
  }
  if (realNums == 0) {
    return true;
  }
  if (gThreadPool == nullptr) {
    gThreadPool = std::unique_ptr<LiteThreadPool>(new (std::nothrow) LiteThreadPool(realNums));
    if (gThreadPool == nullptr) {
      MS_LOGE("%d threads create failed", realNums);
      return false;
    }
  } else {
    gThreadPool->AddNewThread(realNums);
  }
  MS_LOGD("%d threads create successful", realNums);
  return true;
}

ThreadPool *ThreadPool::GetInstance() {
  static ThreadPool instance;
  return &instance;
}

void ThreadPool::ConfigThreadPool(int mode, int numThreads) {
  bindMode = mode;
  totalThreadNum = numThreads;
}

bool ThreadPool::LaunchThreadPoolTask() {
  if (gThreadPool == nullptr) {
    if (!SetThreadPool(totalThreadNum)) {
      MS_LOGE("create %d threads failed", totalThreadNum);
      return false;
    }
  }

  if (gThreadBind == nullptr) {
    if (!SetThreadCpulBind(bindMode)) {
      MS_LOGE("create bind mode %d failed", bindMode);
      return false;
    }
  }
  return true;
}

bool ThreadPool::AddTask(const WorkFun &worker, void *cdata, int numTask) {
  if (numTask <= 0) {
    numTask = totalThreadNum;
  }
  // single task, run master thread
  if (numTask <= 1) {
    TvmEnv env{};
    env.num_task = numTask;
    int ret = worker(0, &env, cdata);
    if (ret != 0) {
      MS_LOGE("task 0 failed, error code is %d", ret);
      return false;
    }
    MS_LOGD("task 0 successful");
    return true;
  }
  ThreadPoolTask task;
  task.first = worker;
  task.second.cdata = cdata;
  return gThreadPool->DistributeTask(task, numTask);
}

LiteThreadPool::~LiteThreadPool() {
  destroy.store(true);
  running.store(0);
  queueReady.notify_all();
  for (auto &thread : threadList) {
    if (thread.joinable()) {
      thread.join();
    }
  }
}
}  // namespace predict
}  // namespace mindspore
