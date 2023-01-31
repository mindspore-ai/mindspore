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
#ifndef AICPU_CONTEXT_COMMON_DEVICE_SHARDER_H
#define AICPU_CONTEXT_COMMON_DEVICE_SHARDER_H
#include "cpu_kernel/common/sharder.h"

namespace aicpu {
using ParallelForFunc = void (*)(int64_t total, int64_t perUnitSize, const std::function<void(int64_t, int64_t)> &work);
using GetCPUNumFunc = uint32_t (*)();
class DeviceSharder : public Sharder {
 public:
  explicit DeviceSharder(DeviceType device);

  ~DeviceSharder() override;

  /*
   * ParallelFor shards the "total" units of work.
   * @param total: size of total work
   * @param perUnitSize: expect size of per unit work
   * @param work: process of per unit work
   */
  void ParallelFor(int64_t total, int64_t perUnitSize,
                   const std::function<void(int64_t, int64_t)> &work) const override;

  /*
   * Get CPU number
   * @return CPU number
   */
  uint32_t GetCPUNum() const override;

 private:
  DeviceSharder(const DeviceSharder &) = delete;
  DeviceSharder(DeviceSharder &&) = delete;
  DeviceSharder &operator=(const DeviceSharder &) = delete;
  DeviceSharder &operator=(DeviceSharder &&) = delete;

 private:
  void *sharder_;
  ParallelForFunc parallel_for_;
  GetCPUNumFunc get_cpu_num_;
};
}  // namespace aicpu
#endif  // AICPU_CONTEXT_COMMON_DEVICE_SHARDER_H
