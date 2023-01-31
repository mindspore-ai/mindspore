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
#ifndef AICPU_CONTEXT_COMMON_SHARDER_H
#define AICPU_CONTEXT_COMMON_SHARDER_H
#include <functional>

#include "cpu_kernel/inc/cpu_types.h"

namespace aicpu {
class Sharder {
 public:
  explicit Sharder(DeviceType device) : device_(device) {}

  virtual ~Sharder() = default;

  /*
   * ParallelFor shards the "total" units of work.
   * @param total: size of total work
   * @param perUnitSize: expect size of per unit work
   * @param work: process of per unit work
   */
  virtual void ParallelFor(int64_t total, int64_t perUnitSize,
                           const std::function<void(int64_t, int64_t)> &work) const = 0;

  /*
   * Get CPU number
   * @return CPU number
   */
  virtual uint32_t GetCPUNum() const = 0;

 private:
  Sharder(const Sharder &) = delete;
  Sharder(Sharder &&) = delete;
  Sharder &operator=(const Sharder &) = delete;
  Sharder &operator=(Sharder &&) = delete;

 private:
  DeviceType device_;  // device type, HOST/DEVICE
};
}  // namespace aicpu
#endif  // AICPU_CONTEXT_COMMON_SHARDER_H
