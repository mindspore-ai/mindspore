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

#ifndef AICPU_OPS_AICPU_SHARDER_H_
#define AICPU_OPS_AICPU_SHARDER_H_

#include <functional>
#include <vector>
#include "common/kernel_util.h"

namespace aicpu {
using Closure = std::function<void()>;
using ClosureBool = std::function<bool()>;
using RunnerBool = std::function<bool(Closure, bool)>;
using SharderWork = std::function<void(int64_t, int64_t)>;

class SharderNonBlock {
 public:
  /**
   * Get the unique object of this class
   */
  static SharderNonBlock &GetInstance();

  /**
   * Register schedule callback function, do_task function and cpu core number
   * called by compute process
   * @param schedule Schedule callback function
   * @param do_task Callback function for itself schedule
   * @param cpu_core_num aicpu core number
   */
  void Register(const RunnerBool &schedule, const ClosureBool &do_task, uint32_t cpu_core_num);

  /**
   * Shards the "total" unit of work refer "perUintSize"
   * @param total Total unit of work
   * @param per_unit_size Minimum shard unit
   * @param work should be a callable taking (int64, int64) arguments.
                 work(start, limit) computes the work units from [start, limit),
                 i.e., [start, limit) is a shard.
   */
  void ParallelFor(int64_t total, int64_t per_unit_size, const SharderWork &work);

  /**
   * Shards the unit of work refer for hash
   * @param total, Total unit of work
   * @param cpu_nums Number of cpu cores
   * @param work should be a callable taking (int64, int64) arguments.
                 work(cur, cpu_nums) computes the work units with input hash with (cpu_nums-1) equals cur,
                 i.e. specially used by parallel unique op
   */
  void ParallelForHash(int64_t total, int64_t cpu_nums, const SharderWork &work);

  /**
   * Schedule a task use schedule function registered by compute process,
   * note that the task will actually executed asynchronously
   * @param closure Closure function with nothrow
   */
  void Schedule(const Closure &closure);

  /**
   * Get CPU number
   * @param None
   * @return CPU number
   */
  uint32_t GetCPUNum();

 private:
  SharderNonBlock() : schedule_(nullptr), do_task_(nullptr), cpu_core_num_(0) {}
  ~SharderNonBlock() = default;

  SharderNonBlock(const SharderNonBlock &) = delete;
  SharderNonBlock &operator=(const SharderNonBlock &) = delete;
  SharderNonBlock(SharderNonBlock &&) = delete;
  SharderNonBlock &operator=(SharderNonBlock &&) = delete;

  /**
   * Closure function enqueue
   * @param closure Closure function can be called
   * @param submit_topic whether submit topic, true means submit topic
   * @return whether enqueue of closure success
   */
  bool Enqueue(const Closure &closure, bool submit_topic = false);

  /**
   * Calculate how many times, which ceiled, "x" is "base".
   * i.e., x is 1, base is 2, this function will return 1
   * @param x An integral
   * @param base An integral as base when cal multiple
   * @return ceiled multiple
   */
  inline int64_t CeilMultiple(int64_t x, int64_t base);

 private:
  RunnerBool schedule_;    // enqueue runner
  ClosureBool do_task_;    // a callback, do task from task queue
  uint32_t cpu_core_num_;  // aicpu core number
};                         // SharderNonBlock
}  // namespace aicpu

extern "C" {
/**
 * Shards the "total" unit of work refer "perUintSize"
 * @param total Total unit of work
 * @param per_unit_size Minimum shard unit
 * @param work should be a callable taking (int64, int64) arguments.
                 work(start, limit) computes the work units from [start, limit),
                i.e., [start, limit) is a shard.
 */
AICPU_VISIBILITY_API void ParallelFor(int64_t total, int64_t per_unit_size, const aicpu::SharderWork &work);

/**
 * Get CPU number
 * @param None
 * @return CPU number
 */
AICPU_VISIBILITY_API uint32_t GetCPUNum();
}

#endif  // AICPU_OPS_AICPU_SHARDER_H_
