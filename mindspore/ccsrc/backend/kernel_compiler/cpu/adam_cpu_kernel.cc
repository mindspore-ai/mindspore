/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "backend/kernel_compiler/cpu/nnacl/errorcode.h"
#include "backend/kernel_compiler/cpu/nnacl/fp32/adam_fp32.h"
#include "runtime/device/cpu/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kAdamInputsNum = 10;
constexpr size_t kAdamOutputsNum = 3;
constexpr size_t kScalarIndex = 0;
}  // namespace

template <typename T>
void AdamCPUKernel::LaunchAdam(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &) {
  T *var = reinterpret_cast<T *>(inputs[VAR]->addr);
  T *m = reinterpret_cast<T *>(inputs[M]->addr);
  T *v = reinterpret_cast<T *>(inputs[V]->addr);
  float beta1_power = reinterpret_cast<float *>(inputs[BETA1_POWER]->addr)[kScalarIndex];
  float beta2_power = reinterpret_cast<float *>(inputs[BETA2_POWER]->addr)[kScalarIndex];
  float lr = reinterpret_cast<float *>(inputs[LR]->addr)[kScalarIndex];
  T beta1 = static_cast<T>(reinterpret_cast<float *>(inputs[BETA1]->addr)[kScalarIndex]);
  T beta2 = static_cast<T>(reinterpret_cast<float *>(inputs[BETA1]->addr)[kScalarIndex]);
  T epsilon = static_cast<T>(reinterpret_cast<float *>(inputs[EPSILON]->addr)[kScalarIndex]);
  T *gradient = reinterpret_cast<T *>(inputs[GRAD]->addr);
  constexpr float ONE = 1.0;
  if (beta1_power - ONE == 0) {
    MS_LOG(EXCEPTION) << "The beta1_power can't be set 1.";
  }
  T new_lr = static_cast<T>(lr * std::sqrt(ONE - beta2_power) / (ONE - beta1_power));
  // multithreading
  size_t lens = inputs[VAR]->size > 0 ? static_cast<size_t>(inputs[VAR]->size / sizeof(T)) : 1;
  auto task = [this, &var, &m, &v, &gradient, new_lr, beta1, beta2, epsilon](size_t start, size_t end) {
    T one = static_cast<T>(1.0);
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
                                    const std::vector<kernel::AddressPtr> &) {
  float *var = reinterpret_cast<float *>(inputs[VAR]->addr);
  float *m = reinterpret_cast<float *>(inputs[M]->addr);
  float *v = reinterpret_cast<float *>(inputs[V]->addr);
  float beta1_power = reinterpret_cast<float *>(inputs[BETA1_POWER]->addr)[kScalarIndex];
  float beta2_power = reinterpret_cast<float *>(inputs[BETA2_POWER]->addr)[kScalarIndex];
  float lr = reinterpret_cast<float *>(inputs[LR]->addr)[kScalarIndex];
  float beta1 = reinterpret_cast<float *>(inputs[BETA1]->addr)[kScalarIndex];
  float beta2 = reinterpret_cast<float *>(inputs[BETA2]->addr)[kScalarIndex];
  float epsilon = reinterpret_cast<float *>(inputs[EPSILON]->addr)[kScalarIndex];
  float *gradient = reinterpret_cast<float *>(inputs[GRAD]->addr);
  constexpr float ONE = 1.0;
  if (beta1_power - ONE == 0) {
    MS_LOG(EXCEPTION) << "The beta1_power can't be set 1.";
  }
  float new_lr = lr * std::sqrt(ONE - beta2_power) / (ONE - beta1_power);

  // multithreading
  size_t lens = inputs[VAR]->size > 0 ? static_cast<size_t>(inputs[VAR]->size / sizeof(float)) : 1;
  auto task = [this, &var, &m, &v, &gradient, new_lr, beta1, beta2, epsilon](size_t start, size_t end) {
    int ret = AdamFp32(var, m, v, new_lr, beta1, beta2, epsilon, gradient, start, end, use_nesterov_);
    if (ret != NNACL_OK) {
      MS_LOG(EXCEPTION) << "AdamFp32 failed.";
    }
  };
  ParallelLaunchAutoSearch(task, lens, this, &parallel_search_info_);
}

void AdamCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  CHECK_KERNEL_INPUTS_NUM(input_num, kAdamInputsNum, kernel_name_);
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kAdamOutputsNum, kernel_name_);
  use_nesterov_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, USE_NESTEROV);
}

bool AdamCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                           const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kAdamInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kAdamOutputsNum, kernel_name_);

  if (inputs[VAR]->size != inputs[M]->size || inputs[VAR]->size != inputs[V]->size ||
      inputs[VAR]->size != inputs[GRAD]->size) {
    MS_LOG(EXCEPTION) << "Error input data size!";
  }
  size_t f_size = sizeof(float);
  if (inputs[BETA1_POWER]->size != f_size || inputs[BETA2_POWER]->size != f_size || inputs[LR]->size != f_size ||
      inputs[BETA1]->size != f_size || inputs[BETA2]->size != f_size || inputs[EPSILON]->size != f_size) {
    MS_LOG(EXCEPTION) << "The attribute beta_power, beta, lr and epsilon must be float!";
  }

  if (dtype_ == kNumberTypeFloat32) {
    LaunchAdamNnacl(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat16) {
    LaunchAdam<float16>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "Adam not support " << dtype_;
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
