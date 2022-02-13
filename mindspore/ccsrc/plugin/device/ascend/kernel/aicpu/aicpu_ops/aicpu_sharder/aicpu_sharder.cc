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
#include "aicpu_sharder/aicpu_sharder.h"

#include <semaphore.h>
#include <unistd.h>
#include <error.h>
#include <atomic>
#include <algorithm>
#include <cerrno>
#include <cstring>

#include "common/kernel_log.h"

namespace aicpu {
#define AICPU_SHARDER_IF_TRUE_RUN(expr, run) \
  do {                                       \
    if (expr) {                              \
      run;                                   \
    }                                        \
  } while (0)

void SharderNonBlock::Register(const RunnerBool &schedule, const ClosureBool &do_task, uint32_t cpu_core_num) {
  schedule_ = schedule;
  do_task_ = do_task;
  cpu_core_num_ = cpu_core_num;
}

bool SharderNonBlock::Enqueue(const Closure &closure, bool submit_topic) {
  if (schedule_ != nullptr) {
    return schedule_(closure, submit_topic);
  }
  return false;
}

void SharderNonBlock::Schedule(const Closure &closure) {
  if (!Enqueue(closure)) {
    closure();
  }
}

uint32_t SharderNonBlock::GetCPUNum() { return cpu_core_num_; }

SharderNonBlock &SharderNonBlock::GetInstance() {
  static SharderNonBlock sharder_non_block;
  return sharder_non_block;
}

int64_t SharderNonBlock::CeilMultiple(int64_t x, int64_t base) {
  if (base == 0) {
    return 0;
  }
  int64_t ret = x / base;
  if ((x % base) != 0) {
    ret++;
  }

  return ret;
}

void SharderNonBlock::ParallelFor(int64_t total, int64_t per_unit_size, const SharderWork &work) {
  AICPU_LOGI("total: %lld, per_unit_size: %lld", total, per_unit_size);
  if ((total <= 0) || (work == nullptr)) {
    AICPU_LOGE("invalid param: total<=0 or work is nullptr");
    return;
  }

  // work itself
  if ((schedule_ == nullptr) || (cpu_core_num_ <= 1)) {
    AICPU_LOGI("work itself all");
    work(0, total);
    return;
  }

  // In order to ensure a smaller scheduling delay, the maximum number of slices is twice the number of CPU cores
  const int64_t max_shard_num = static_cast<int64_t>(cpu_core_num_) * 2;

  // calculate shard number and block size
  // i.e., if total is 118, perUintSize is 2, and cpu_core_num_ is 13
  // then shard_num is 24, block_size is 5
  int64_t block_size = std::max(int64_t{1}, std::min(total, per_unit_size));
  int64_t shard_num = CeilMultiple(total, block_size);
  shard_num = std::min(max_shard_num, shard_num);
  block_size = CeilMultiple(total, shard_num);
  shard_num = CeilMultiple(total, block_size);
  AICPU_LOGI("shard number: %lld, block size: %lld", shard_num, block_size);

  // There is no need to submit an event if shard_num is 1
  if (shard_num == 1) {
    AICPU_LOGI("executes on the current thread");
    work(0, total);
    return;
  }

  std::atomic<int64_t> count(shard_num);  // a counter
  sem_t sem;
  int32_t sem_init_ret = sem_init(&sem, 0, 0);
  if (sem_init_ret == -1) {
    AICPU_LOGE("sem_init error with message: %s", strerror(errno));
    work(0, total);
    return;
  }

  for (int64_t start = 0; start < total; start += block_size) {
    auto limit = std::min(start + block_size, total);
    Closure closure = [&sem, &work, &count, start, limit]() {
      count--;
      // In order to ensure that user's work function exception does not affect multithread services,
      // exception capture is needed. Exception type is not cared here, and error log is printed.
      try {
        work(start, limit);
      } catch (...) {
        AICPU_LOGE("exception occurred in work function with start: %lld, limit: %lld", start, limit);
      }

      int32_t sem_post_ret = sem_post(&sem);
      AICPU_SHARDER_IF_TRUE_RUN(sem_post_ret == -1, AICPU_LOGE("sem_post error with message: %s", strerror(errno)));
    };

    // if enqueue fail, work itself
    if (!Enqueue(closure, true)) {
      AICPU_LOGI("Enqueue fail, [%lld, %lld), work itself", start, limit);
      closure();
    }
  }

  if (do_task_ != nullptr) {
    bool ret = true;
    while ((count > 0) && ret) {
      AICPU_LOGI("Main thread do task begin.");
      ret = do_task_();
      AICPU_LOGI("Main thread do task end.");
    }
  }

  for (int64_t i = 0; i < shard_num; ++i) {
    int sem_wait_ret = sem_wait(&sem);
    AICPU_SHARDER_IF_TRUE_RUN(sem_wait_ret == -1, AICPU_LOGE("sem_wait error with message: %s", strerror(errno)));
  }
  int32_t sem_des_ret = sem_destroy(&sem);
  AICPU_SHARDER_IF_TRUE_RUN(sem_des_ret == -1, AICPU_LOGE("sem_destroy error with message: %s", strerror(errno)));
}

void SharderNonBlock::ParallelForHash(int64_t total, int64_t cpu_nums, const SharderWork &work) {
  AICPU_LOGI("total: %lld, cpu_nums: %d", total, cpu_nums);
  if (total <= 0 || work == nullptr) {
    AICPU_LOGE("invalid param: total<=0 or work is nullptr");
    return;
  }

  if ((schedule_ == nullptr) || (cpu_core_num_ <= 1)) {
    AICPU_LOGE("schedule is nullptr or cpu core num is not enough");
    return;
  }

  std::atomic<int64_t> count(cpu_nums);  // a counter

  sem_t sem;
  int32_t sem_init_ret = sem_init(&sem, 0, 0);
  if (sem_init_ret == -1) {
    AICPU_LOGE("sem_init error with message: %s", strerror(errno));
    return;
  }

  for (int64_t cur = 0; cur < cpu_nums; cur++) {
    Closure closure = [&sem, &work, &count, total, cur]() {
      work(total, cur);
      count--;
      int32_t sem_post_ret = sem_post(&sem);
      AICPU_SHARDER_IF_TRUE_RUN(sem_post_ret == -1, AICPU_LOGE("sem_post error with message: %s", strerror(errno)));
    };

    // if enqueue fail, work itself
    if (!Enqueue(closure, true)) {
      closure();
    }
  }

  if (do_task_ != nullptr) {
    bool ret = true;
    while ((count > 0) && ret) {
      ret = do_task_();
    }
  }

  for (int64_t i = 0; i < cpu_nums; i++) {
    int sem_wait_ret = sem_wait(&sem);
    AICPU_SHARDER_IF_TRUE_RUN(sem_wait_ret == -1, AICPU_LOGE("sem_wait error with message: %s", strerror(errno)));
  }
  int32_t sem_des_ret = sem_destroy(&sem);
  AICPU_SHARDER_IF_TRUE_RUN(sem_des_ret == -1, AICPU_LOGE("sem_destroy error with message: %s", strerror(errno)));
}
}  // namespace aicpu

/**
 * Shards the "total" unit of work refer "perUintSize"
 */
void ParallelFor(int64_t total, int64_t per_unit_size, const aicpu::SharderWork &work) {
  aicpu::SharderNonBlock::GetInstance().ParallelFor(total, per_unit_size, work);
}

/**
 * Get CPU number
 */
uint32_t GetCPUNum() { return aicpu::SharderNonBlock::GetInstance().GetCPUNum(); }
