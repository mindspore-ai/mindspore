/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/adam_weight_decay_cpu_kernel.h"

#include <cmath>

#include "backend/kernel_compiler/cpu/nnacl/errorcode.h"
#include "backend/kernel_compiler/cpu/nnacl/fp32/adam_fp32.h"
#include "runtime/device/cpu/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSizeFloat32 = sizeof(float);
constexpr size_t kScalarIndex = 0;
constexpr size_t kAdamWeightDecayInputsNum = 9;
constexpr size_t kAdamWeightDecayOutputsNum = 3;
}  // namespace

template <typename T>
void AdamWeightDecayCPUKernel::LaunchAdamWeightDecay(const std::vector<AddressPtr> &inputs,
                                                     const std::vector<AddressPtr> &) {
  T *var = reinterpret_cast<T *>(inputs[VAR]->addr);
  T *m = reinterpret_cast<T *>(inputs[M]->addr);
  T *v = reinterpret_cast<T *>(inputs[V]->addr);
  T lr = static_cast<T>(reinterpret_cast<float *>(inputs[LR]->addr)[kScalarIndex]);
  T beta1 = static_cast<T>(reinterpret_cast<float *>(inputs[BETA1]->addr)[kScalarIndex]);
  T beta2 = static_cast<T>(reinterpret_cast<float *>(inputs[BETA2]->addr)[kScalarIndex]);
  T epsilon = static_cast<T>(reinterpret_cast<float *>(inputs[EPSILON]->addr)[kScalarIndex]);
  T decay = static_cast<T>(reinterpret_cast<float *>(inputs[DECAY]->addr)[kScalarIndex]);
  T *gradient = reinterpret_cast<T *>(inputs[GRAD]->addr);
  const T one = static_cast<T>(1.0);
  const T beta1_minus = one - beta1;
  const T beta2_minus = one - beta2;

  // multithreading
  size_t lens = inputs[VAR]->size > 0 ? static_cast<size_t>(inputs[VAR]->size / sizeof(T)) : 1;
  std::function<void(size_t, size_t)> task;
  task = [&](size_t start, size_t end) {
    // remaining
    for (size_t i = start; i < end; i++) {
      m[i] += (gradient[i] - m[i]) * beta1_minus;
      v[i] += (gradient[i] * gradient[i] - v[i]) * beta2_minus;
      T sqrt_v = static_cast<T>(std::sqrt(static_cast<double>(v[i])));
      auto update = m[i] / (sqrt_v + epsilon) + decay * var[i];
      var[i] -= lr * update;
    }
  };
  CPUKernelUtils::ParallelFor(task, lens);
}

void AdamWeightDecayCPUKernel::LaunchAdamWeightDecayNnacl(const std::vector<AddressPtr> &inputs,
                                                          const std::vector<AddressPtr> &) {
  auto var = reinterpret_cast<float *>(inputs[VAR]->addr);
  auto m = reinterpret_cast<float *>(inputs[M]->addr);
  auto v = reinterpret_cast<float *>(inputs[V]->addr);
  auto lr = reinterpret_cast<float *>(inputs[LR]->addr)[kScalarIndex];
  auto beta1 = reinterpret_cast<float *>(inputs[BETA1]->addr)[kScalarIndex];
  auto beta2 = reinterpret_cast<float *>(inputs[BETA2]->addr)[kScalarIndex];
  auto epsilon = reinterpret_cast<float *>(inputs[EPSILON]->addr)[kScalarIndex];
  auto decay = reinterpret_cast<float *>(inputs[DECAY]->addr)[kScalarIndex];
  auto gradient = reinterpret_cast<float *>(inputs[GRAD]->addr);

  // multithreading
  size_t lens = inputs[VAR]->size > 0 ? static_cast<size_t>(inputs[VAR]->size / sizeof(float)) : 1;
  std::function<void(size_t, size_t)> task;
  task = [&](size_t start, size_t end) {
    int ret = AdamWeightDecayFp32(var, m, v, lr, beta1, beta2, epsilon, decay, gradient, start, end);
    if (ret != NNACL_OK) {
      MS_LOG(EXCEPTION) << "AdamWeightDecayFp32 failed.";
    }
  };
  ParallelLaunchAutoSearch(task, lens, this, &parallel_search_info_);
}

void AdamWeightDecayCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
}

bool AdamWeightDecayCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                      const std::vector<kernel::AddressPtr> &,
                                      const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kAdamWeightDecayInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kAdamWeightDecayOutputsNum, kernel_name_);
  if (inputs[VAR]->size != inputs[M]->size || inputs[VAR]->size != inputs[V]->size ||
      inputs[VAR]->size != inputs[GRAD]->size) {
    MS_LOG(EXCEPTION) << "Var, m, v, grad input data size must be same!";
  }
  if (inputs[LR]->size != kSizeFloat32 || inputs[BETA1]->size != kSizeFloat32 || inputs[BETA2]->size != kSizeFloat32 ||
      inputs[EPSILON]->size != kSizeFloat32 || inputs[DECAY]->size != kSizeFloat32) {
    MS_LOG(EXCEPTION) << "The attribute beta, lr, epsilon and weight decay must be float!";
  }

  if (dtype_ == kNumberTypeFloat32) {
    LaunchAdamWeightDecayNnacl(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat16) {
    LaunchAdamWeightDecay<float16>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "AdamWeightDecay not support " << dtype_;
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
