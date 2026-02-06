/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include <atomic>
#include <limits>
#include <thread>
#include <vector>

#include "common/common_test.h"
#include "include/runtime/utils/runtime_conf/runtime_conf.h"
#include "kernel/cpu/cpu_kernel.h"
#include "thread/threadpool.h"

namespace mindspore {
namespace kernel {
namespace {
class RecordingThreadPool : public mindspore::ThreadPool {
 public:
  RecordingThreadPool() = default;
  ~RecordingThreadPool() override = default;

  int ParallelLaunch(const mindspore::Func &func, mindspore::Content content, int task_num) override {
    last_task_num_.store(task_num);
    for (int task_id = 0; task_id < task_num; ++task_id) {
      (void)func(content, task_id, 0.f, 0.f);
    }
    return 0;
  }

  int last_task_num() const { return last_task_num_.load(); }

 private:
  std::atomic<int> last_task_num_{-1};
};

class CpuKernelParallelTaskLimitTest : public UT::Common {
 public:
  void TearDown() override {
    // RuntimeConf is a process-level singleton; reset to a "no-op" large value to avoid
    // affecting other unit tests in the same binary.
    auto conf = runtime::RuntimeConf::GetInstance();
    if (conf != nullptr) {
      conf->set_op_threads_num(std::numeric_limits<uint32_t>::max());
    }
  }
};
}  // namespace

/// Feature: CPU kernel ParallelLaunch task count is capped by RuntimeConf::op_threads_num.
/// Description: Set op_threads_num=2 and run ParallelLaunch(CTask) with a ThreadPool whose kernel threads > op_threads_num.
/// Expectation: ParallelLaunch uses at most 2 tasks (task_num == 2).
TEST_F(CpuKernelParallelTaskLimitTest, ParallelLaunchTaskCountCappedByOpThreadsNum) {
  auto conf = runtime::RuntimeConf::GetInstance();
  ASSERT_TRUE(conf != nullptr);
  conf->set_op_threads_num(2);

  RecordingThreadPool pool;
  pool.SetActorThreadNum(0);
  pool.SetKernelThreadNum(8);  // larger than op_threads_num

  std::atomic<int> counter{0};
  CTask task = [&counter](size_t, size_t) { counter.fetch_add(1); };

  ParallelLaunch(task, /*count*/ 100, /*block_size*/ 1.0f, /*content*/ nullptr, /*pool*/ &pool);
  EXPECT_LE(pool.last_task_num(), 2);
  EXPECT_EQ(pool.last_task_num(), 2);
}

/// Feature: CPU kernel ParallelLaunch(vector<Task>) task count is capped by RuntimeConf::op_threads_num.
/// Description: Set op_threads_num=3 and launch a vector of many tasks using a ThreadPool whose kernel threads > op_threads_num.
/// Expectation: ParallelLaunch requests no more than 3 tasks from the threadpool (task_num <= 3).
TEST_F(CpuKernelParallelTaskLimitTest, ParallelLaunchVectorTasksCappedByOpThreadsNum) {
  auto conf = runtime::RuntimeConf::GetInstance();
  ASSERT_TRUE(conf != nullptr);
  conf->set_op_threads_num(3);

  RecordingThreadPool pool;
  pool.SetActorThreadNum(0);
  pool.SetKernelThreadNum(8);  // larger than op_threads_num

  std::atomic<int> counter{0};
  std::vector<common::Task> tasks;
  tasks.reserve(100);
  for (size_t i = 0; i < 100; ++i) {
    tasks.emplace_back([&counter]() {
      counter.fetch_add(1);
      return common::SUCCESS;
    });
  }

  ParallelLaunch(tasks, /*content*/ nullptr, /*pool*/ &pool);
  EXPECT_LE(pool.last_task_num(), 3);
}

/// Feature: CPU kernel ParallelLaunch single-task fast-path runs on the caller (master) thread.
/// Description: Set op_threads_num=1 and run ParallelLaunch(CTask) using a real MindRT ThreadPool; the task records the running thread id.
/// Expectation: Task executes on the caller thread (master-thread SyncRunFunc path), i.e., recorded thread id equals caller thread id.
TEST_F(CpuKernelParallelTaskLimitTest, ParallelLaunchSingleTaskRunsOnMasterThread) {
  auto conf = runtime::RuntimeConf::GetInstance();
  ASSERT_TRUE(conf != nullptr);
  conf->set_op_threads_num(1);

  // Use the real MindRT threadpool to validate the master-thread fast-path:
  // ThreadPool::ParallelLaunch does SyncRunFunc when task_num <= 1.
  auto pool = mindspore::ThreadPool::CreateThreadPool(2);
  ASSERT_TRUE(pool != nullptr);
  pool->SetActorThreadNum(0);
  pool->SetKernelThreadNum(2);

  const auto caller_tid = std::this_thread::get_id();
  std::atomic<bool> ran_on_master{false};
  CTask task = [&](size_t, size_t) { ran_on_master.store(std::this_thread::get_id() == caller_tid); };

  // Ensure cpu_kernel.cc computes task_num == 1 (since op_threads_num == 1).
  ParallelLaunch(task, /*count*/ 128, /*block_size*/ 1.0f, /*content*/ nullptr, /*pool*/ pool);
  EXPECT_TRUE(ran_on_master.load());

  delete pool;
}

}  // namespace kernel
}  // namespace mindspore
