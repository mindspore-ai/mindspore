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
#include <cmath>
#include "backend/kernel_compiler/cpu/mkldnn/mkl_kernel_engine.h"
#include "runtime/device/cpu/cpu_device_address.h"
#include "backend/kernel_compiler/cpu/adam_cpu_kernel.h"
#include "nnacl/errorcode.h"
#include "nnacl/fp32/adam_fp32.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
template <typename T>
void AdamCPUKernel::LaunchAdam(const std::vector<kernel::AddressPtr> &inputs,
                               const std::vector<kernel::AddressPtr> &outputs) {
  T *var = reinterpret_cast<T *>(inputs[0]->addr);
  T *m = reinterpret_cast<T *>(inputs[1]->addr);
  T *v = reinterpret_cast<T *>(inputs[2]->addr);
  float beta1_power = reinterpret_cast<float *>(inputs[3]->addr)[0];
  float beta2_power = reinterpret_cast<float *>(inputs[4]->addr)[0];
  float lr = reinterpret_cast<float *>(inputs[5]->addr)[0];
  T beta1 = static_cast<T>(reinterpret_cast<float *>(inputs[6]->addr)[0]);
  T beta2 = static_cast<T>(reinterpret_cast<float *>(inputs[7]->addr)[0]);
  T epsilon = static_cast<T>(reinterpret_cast<float *>(inputs[8]->addr)[0]);
  T *gradient = reinterpret_cast<T *>(inputs[9]->addr);
  if (beta1_power - 1.0 == 0) {
    MS_LOG(EXCEPTION) << "The beta1_power can't be set 1.";
  }
  T new_lr = static_cast<T>(lr * std::sqrt(1.0 - beta2_power) / (1 - beta1_power));
  T one = static_cast<T>(1.0);
  // multithreading
  size_t lens = inputs[0]->size > 0 ? static_cast<size_t>(inputs[0]->size / sizeof(T)) : 1;
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      m[i] += (gradient[i] - m[i]) * (one - beta1);
      v[i] += (gradient[i] * gradient[i] - v[i]) * (one - beta2);
      T sqrt_v = static_cast<T>(std::sqrt(static_cast<double>(v[i])));
      if (use_nesterov_) {
        var[i] -= new_lr * (m[i] * beta1 + (one - beta1) * gradient[i]) / (sqrt_v + epsilon);
      } else {
        var[i] -= new_lr * m[i] / (sqrt_v + epsilon);
      }
    }
  };
  CPUKernelUtils::ParallelFor(task, lens);
}

void AdamCPUKernel::LaunchAdamNnacl(const std::vector<kernel::AddressPtr> &inputs,
                                    const std::vector<kernel::AddressPtr> &outputs) {
  float *var = reinterpret_cast<float *>(inputs[0]->addr);
  float *m = reinterpret_cast<float *>(inputs[1]->addr);
  float *v = reinterpret_cast<float *>(inputs[2]->addr);
  float beta1_power = reinterpret_cast<float *>(inputs[3]->addr)[0];
  float beta2_power = reinterpret_cast<float *>(inputs[4]->addr)[0];
  float lr = reinterpret_cast<float *>(inputs[5]->addr)[0];
  float beta1 = reinterpret_cast<float *>(inputs[6]->addr)[0];
  float beta2 = reinterpret_cast<float *>(inputs[7]->addr)[0];
  float epsilon = reinterpret_cast<float *>(inputs[8]->addr)[0];
  float *gradient = reinterpret_cast<float *>(inputs[9]->addr);
  if (beta1_power - 1.0 == 0) {
    MS_LOG(EXCEPTION) << "The beta1_power can't be set 1.";
  }
  float new_lr = lr * std::sqrt(1.0 - beta2_power) / (1 - beta1_power);

  // multithreading
  size_t lens = inputs[0]->size > 0 ? static_cast<size_t>(inputs[0]->size / sizeof(float)) : 1;
  auto task = [&](size_t start, size_t end) {
    AdamFp32(var, m, v, new_lr, beta1, beta2, epsilon, gradient, start, end, use_nesterov_);
  };
  CPUKernelUtils::ParallelFor(task, lens);
}

void AdamCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  if (input_num != 10) {
    MS_LOG(EXCEPTION) << "Input number is " << input_num << ", but Adam needs 10 inputs.";
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 3) {
    MS_LOG(EXCEPTION) << "Output number is " << output_num << ", but Adam needs 3 outputs.";
  }
  use_nesterov_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, "use_nesterov");
}

bool AdamCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
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

  if (dtype_ == kNumberTypeFloat32) {
    LaunchAdamNnacl(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat16) {
    LaunchAdam<float16>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "Adam not support " << dtype_;
    return false;
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
