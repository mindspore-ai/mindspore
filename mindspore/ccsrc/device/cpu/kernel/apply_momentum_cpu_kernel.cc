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
#include "device/cpu/kernel/apply_momentum_cpu_kernel.h"
#include "device/cpu/kernel/mkldnn/mkl_kernel_engine.h"
#include "device/cpu/cpu_device_address.h"
#include "common/utils.h"

namespace mindspore {
namespace device {
namespace cpu {
void ApplyMomentumCPUKernel::InitKernel(const CNodePtr & /*kernel_node*/) {}

bool ApplyMomentumCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                    const std::vector<kernel::AddressPtr> & /*workspace*/,
                                    const std::vector<kernel::AddressPtr> & /*outputs*/) {
  if (inputs.size() < 5) {
    MS_LOG(EXCEPTION) << "error input output size!";
  }
  if (inputs[0]->size != inputs[1]->size || inputs[0]->size != inputs[3]->size) {
    MS_LOG(EXCEPTION) << "error input data size!";
  }
  auto weight = reinterpret_cast<float *>(inputs[0]->addr);
  auto accumulate = reinterpret_cast<float *>(inputs[1]->addr);
  float learning_rate = reinterpret_cast<float *>(inputs[2]->addr)[0];
  auto gradient = reinterpret_cast<float *>(inputs[3]->addr);
  float moment = reinterpret_cast<float *>(inputs[4]->addr)[0];
  size_t elem_num = inputs[0]->size / sizeof(float);
  for (size_t i = 0; i < elem_num; ++i) {
    accumulate[i] = accumulate[i] * moment + gradient[i];
    weight[i] -= accumulate[i] * learning_rate;
  }
  return true;
}
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
