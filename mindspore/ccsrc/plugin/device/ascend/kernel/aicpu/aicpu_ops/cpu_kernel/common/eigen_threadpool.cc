/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
#include "cpu_kernel/common/eigen_threadpool.h"

#include <sys/sysinfo.h>

#include "mindspore/ccsrc/plugin/device/ascend/kernel/aicpu/aicpu_ops/common/kernel_log.h"

namespace {
const uint32_t kTaskSize = 40000;
const uint32_t kMaxOverShardingFactor = 4;
const uint32_t kTotalCostFactor = 210000;
constexpr uint32_t kMaxTaskSize = kTaskSize * kMaxOverShardingFactor;
}  // namespace

namespace aicpu {
std::mutex EigenThreadPool::mutex_;
bool EigenThreadPool::init_flag_(false);
int32_t EigenThreadPool::core_num_(0);
std::unique_ptr<Eigen::ThreadPool> EigenThreadPool::eigen_threadpool_(nullptr);
std::unique_ptr<Eigen::ThreadPoolDevice> EigenThreadPool::threadpool_device_(nullptr);

EigenThreadPool *EigenThreadPool::GetInstance() {
  KERNEL_LOG_INFO("EigenThreadPool GetInstance begin");
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!init_flag_) {
      core_num_ = get_nprocs();  // obtains the number of CPU cores that can be
                                 // used by users.
      if (core_num_ <= 0) {
        KERNEL_LOG_INFO(
          "Get the number of CPU cores that can be used failed, core "
          "number[%d]",
          core_num_);
        return nullptr;
      }
      eigen_threadpool_.reset(new Eigen::ThreadPool(core_num_));
      threadpool_device_.reset(new Eigen::ThreadPoolDevice(eigen_threadpool_.get(), core_num_));
      init_flag_ = true;
      KERNEL_LOG_INFO("EigenThreadPool init success, core number[%d]", core_num_);
    }
  }

  static EigenThreadPool instance;
  KERNEL_LOG_INFO("EigenThreadPool GetInstance success");
  return &instance;
}

void EigenThreadPool::ParallelFor(int64_t total, int64_t per_unit_size, const SharderWork &work) const {
  KERNEL_LOG_INFO("Eigen threadpool parallel for begin, total[%ld], per_unit_size[%ld]", total, per_unit_size);
  if ((total <= 0) || (work == nullptr) || (per_unit_size <= 0)) {
    KERNEL_LOG_ERROR(
      "Invalid param: total[%ld] <= 0 or per_unit_size[%ld] <= 0 or work "
      "is "
      "nullptr",
      total, per_unit_size);
    return;
  }

  int64_t total_check = static_cast<int64_t>(static_cast<Eigen::Index>(total));
  if (total_check != total) {
    KERNEL_LOG_ERROR("Invalid param: total[%ld], value[%ld] after eigen conversion", total, total_check);
    return;
  }

  double per_unit_cost = 1.0;
  if (per_unit_size >= total) {
    // use the current thread to process the task
    per_unit_cost = 1.0 * kTaskSize / total;
  } else if ((per_unit_size) <= (total / core_num_)) {
    // run tasks with the maximum number of threads, maximum =
    // kMaxOverShardingFactor * core_num_
    per_unit_cost = (1.0 * kMaxTaskSize * core_num_ / total) > (1.0 * kTotalCostFactor / total)
                      ? (1.0 * kMaxTaskSize * core_num_ / total)
                      : (1.0 * kTotalCostFactor / total);
  } else {
    // the task is fragmented based on the number of data slices.
    per_unit_cost = 1.0 * kMaxTaskSize / per_unit_size;
  }

  KERNEL_LOG_INFO("Eigen threadpool parallel for, per_unit_cost[%.6f]", per_unit_cost);

  threadpool_device_->parallelFor(total, Eigen::TensorOpCost(0, 0, per_unit_cost),
                                  [&work](Eigen::Index first, Eigen::Index last) { work(first, last); });
  KERNEL_LOG_INFO("Eigen threadpool parallel for success");
}

/*
 * Get CPU number
 */
uint32_t EigenThreadPool::GetCPUNum() const { return static_cast<uint32_t>(core_num_); }
}  // namespace aicpu
