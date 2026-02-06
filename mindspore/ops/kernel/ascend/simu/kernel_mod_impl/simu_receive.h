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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_SIMU_SIMU_RECEIVE_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_SIMU_SIMU_RECEIVE_H_

#include <memory>
#include <vector>
#include "kernel/ascend/simu/kernel_mod_impl/simu_kernel.h"

namespace mindspore {
namespace kernel {
class SimuReceiveKernel : public SimuKernel {
 public:
  SimuReceiveKernel() = default;
  ~SimuReceiveKernel() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

 private:
  std::vector<float> init_value_;
  std::vector<float> host_data_;
};

MS_SIMU_REG_KERNEL(Receive, SimuReceiveKernel);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_SIMU_SIMU_RECEIVE_H_
