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
#include "backend/kernel_compiler/cpu/adam_cpu_kernel.h"

#include <cmath>
#include "backend/kernel_compiler/cpu/mkldnn/mkl_kernel_engine.h"
#include "runtime/device/cpu/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
template <typename T>
void AdamCPUKernel::LaunchAdam(T *var, T *m, T *v, float lr, float beta1, float beta2, float epsilon, const T *gradient,
                               size_t size) {
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      m[i] += (gradient[i] - m[i]) * (1 - beta1);
      v[i] += (gradient[i] * gradient[i] - v[i]) * (1 - beta2);
      if (use_nesterov) {
        var[i] -= lr * (m[i] * beta1 + (1 - beta1) * gradient[i]) / (std::sqrt(v[i]) + epsilon);
      } else {
        var[i] -= lr * m[i] / (std::sqrt(v[i]) + epsilon);
      }
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}

void AdamCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 10) {
    MS_LOG(EXCEPTION) << "Input number is " << input_num << ", but Adam needs 10 inputs.";
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 3) {
    MS_LOG(EXCEPTION) << "Output number is " << output_num << ", but Adam needs 3 outputs.";
  }
  use_nesterov = AnfAlgo::GetNodeAttr<bool>(kernel_node, "use_nesterov");
}

bool AdamCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                           const std::vector<kernel::AddressPtr> & /*workspace*/,
                           const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.size() != 10) {
    MS_LOG(EXCEPTION) << "Input number is " << inputs.size() << ", but Adam needs 10 inputs.";
  }
  if (outputs.size() != 3) {
    MS_LOG(EXCEPTION) << "Output number is " << outputs.size() << ", but Adam needs 3 outputs.";
  }
  if (inputs[0]->size != inputs[1]->size || inputs[0]->size != inputs[2]->size || inputs[0]->size != inputs[9]->size) {
    MS_LOG(EXCEPTION) << "Error input data size!";
  }
  size_t f_size = sizeof(float);
  if (inputs[3]->size != f_size || inputs[4]->size != f_size || inputs[5]->size != f_size ||
      inputs[6]->size != f_size || inputs[7]->size != f_size || inputs[8]->size != f_size) {
    MS_LOG(EXCEPTION) << "The attribute beta_power, beta, lr and epsilon must be float!";
  }
  auto var = reinterpret_cast<float *>(inputs[0]->addr);
  auto m = reinterpret_cast<float *>(inputs[1]->addr);
  auto v = reinterpret_cast<float *>(inputs[2]->addr);
  float beta1_power = reinterpret_cast<float *>(inputs[3]->addr)[0];
  float beta2_power = reinterpret_cast<float *>(inputs[4]->addr)[0];
  float lr = reinterpret_cast<float *>(inputs[5]->addr)[0];
  float beta1 = reinterpret_cast<float *>(inputs[6]->addr)[0];
  float beta2 = reinterpret_cast<float *>(inputs[7]->addr)[0];
  float epsilon = reinterpret_cast<float *>(inputs[8]->addr)[0];
  auto gradient = reinterpret_cast<float *>(inputs[9]->addr);
  if (beta1_power == 1) {
    MS_LOG(EXCEPTION) << "The beta1_power can't be set 1.";
  }
  float new_lr = lr * std::sqrt(1.0 - beta2_power) / (1 - beta1_power);

  // multithreading
  size_t lens = inputs[0]->size > 0 ? static_cast<size_t>(inputs[0]->size / sizeof(float)) : 1;
  LaunchAdam<float>(var, m, v, new_lr, beta1, beta2, epsilon, gradient, lens);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
