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
#include "cpu_kernel/common/host_sharder.h"

#include "cpu_kernel/common/eigen_threadpool.h"
#include "mindspore/ccsrc/plugin/device/ascend/kernel/aicpu/aicpu_ops/common/kernel_log.h"

namespace aicpu {
/*
 * ParallelFor shards the "total" units of work.
 */
void HostSharder::ParallelFor(int64_t total, int64_t perUnitSize,
                              const std::function<void(int64_t, int64_t)> &work) const {
  EigenThreadPool *threadpool = EigenThreadPool::GetInstance();
  if (threadpool == nullptr) {
    KERNEL_LOG_ERROR("Get eigen thread pool failed");
    return;
  }

  threadpool->ParallelFor(total, perUnitSize, work);
}

/*
 * Get CPU number
 */
uint32_t HostSharder::GetCPUNum() const {
  EigenThreadPool *threadpool = EigenThreadPool::GetInstance();
  if (threadpool == nullptr) {
    KERNEL_LOG_ERROR("Get eigen thread pool failed");
    return 0;
  }

  return threadpool->GetCPUNum();
}
}  // namespace aicpu
