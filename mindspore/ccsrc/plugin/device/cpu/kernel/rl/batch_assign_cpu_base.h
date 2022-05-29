/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KEREL_RL_BATCH_ASSIGN_CPU_BASE_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KEREL_RL_BATCH_ASSIGN_CPU_BASE_H_

#include <shared_mutex>
#include "plugin/device/cpu/kernel/cpu_kernel.h"

namespace mindspore {
namespace kernel {
class BatchAssignCpuBaseMod : public DeprecatedNativeCpuKernelMod {
 public:
  BatchAssignCpuBaseMod() = default;
  ~BatchAssignCpuBaseMod() override = default;

 protected:
  // Using shared_mutex to achieve the followings：
  // The read-write lock can only have one writer or multiple readers at the same time,
  // but it can't have both readers and writers at the same time.
  static std::shared_mutex rw_mutex_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KEREL_RL_BATCH_ASSIGN_CPU_BASE_H_
