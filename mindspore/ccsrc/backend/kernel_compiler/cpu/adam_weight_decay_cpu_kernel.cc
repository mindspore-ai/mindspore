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
#include "backend/kernel_compiler/cpu/mkldnn/mkl_kernel_engine.h"
#include "runtime/device/cpu/cpu_device_address.h"
#include "nnacl/fp32/adam_fp32.h"
#include "utils/ms_utils.h"
#include "common/thread_pool.h"

namespace mindspore {
namespace kernel {
constexpr size_t kSizeFloat16 = sizeof(float16);
constexpr size_t kSizeFloat32 = sizeof(float);
constexpr size_t kScalarIndex = 0;
constexpr size_t kAdamWeightDecayInputSize = 9;
constexpr size_t kAdamWeightDecayOutputSize = 3;

template <typename T, typename S>
void AdamWeightDecayCPUKernel::LaunchFusedAdam(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &) {
  auto var = reinterpret_cast<T *>(inputs[VAR]->addr);
  auto m = reinterpret_cast<T *>(inputs[M]->addr);
  auto v = reinterpret_cast<T *>(inputs[V]->addr);
  auto lr = reinterpret_cast<T *>(inputs[LR]->addr)[kScalarIndex];
  auto beta1 = reinterpret_cast<T *>(inputs[BETA1]->addr)[kScalarIndex];
  auto beta2 = reinterpret_cast<T *>(inputs[BETA2]->addr)[kScalarIndex];
  auto epsilon = reinterpret_cast<T *>(inputs[EPSILON]->addr)[kScalarIndex];
  auto decay = reinterpret_cast<T *>(inputs[DECAY]->addr)[kScalarIndex];
  auto gradient16 = reinterpret_cast<S *>(inputs[GRAD]->addr);
  const auto beta1_minus = 1 - beta1;
  const auto beta2_minus = 1 - beta2;

  // multithreading
  size_t lens = inputs[VAR]->size > 0 ? static_cast<size_t>(inputs[VAR]->size / sizeof(float)) : 1;
  std::function<void(size_t, size_t)> task;

  task = [&](size_t start, size_t end) {
    size_t i =
      FusedAdamFp32(var, m, v, lr, beta1, beta2, epsilon, decay, reinterpret_cast<int16_t *>(gradient16), start, end);
    // remaining
    for (; i < end; i++) {
      auto temp = static_cast<float>(gradient16[i]);
      m[i] += (temp - m[i]) * beta1_minus;
      v[i] += (temp * temp - v[i]) * beta2_minus;
      T update = m[i] / (std::sqrt(v[i]) + epsilon);
      update += decay * var[i];
      var[i] -= lr * update;
    }
  };
  CPUKernelUtils::ParallelFor(task, lens);
}

template <typename T>
void AdamWeightDecayCPUKernel::LaunchAdamWeightDecay(const std::vector<AddressPtr> &inputs,
                                                     const std::vector<AddressPtr> &) {
  auto var = reinterpret_cast<T *>(inputs[VAR]->addr);
  auto m = reinterpret_cast<T *>(inputs[M]->addr);
  auto v = reinterpret_cast<T *>(inputs[V]->addr);
  auto lr = reinterpret_cast<T *>(inputs[LR]->addr)[kScalarIndex];
  auto beta1 = reinterpret_cast<T *>(inputs[BETA1]->addr)[kScalarIndex];
  auto beta2 = reinterpret_cast<T *>(inputs[BETA2]->addr)[kScalarIndex];
  auto epsilon = reinterpret_cast<T *>(inputs[EPSILON]->addr)[kScalarIndex];
  auto decay = reinterpret_cast<T *>(inputs[DECAY]->addr)[kScalarIndex];
  auto gradient = reinterpret_cast<T *>(inputs[GRAD]->addr);
  const auto beta1_minus = 1 - beta1;
  const auto beta2_minus = 1 - beta2;

  // multithreading
  size_t lens = inputs[VAR]->size > 0 ? static_cast<size_t>(inputs[VAR]->size / sizeof(float)) : 1;
  std::function<void(size_t, size_t)> task;
  task = [&](size_t start, size_t end) {
    size_t i = AdamWeightDecayFp32(var, m, v, lr, beta1, beta2, epsilon, decay, gradient, start, end);
    // remaining
    for (; i < end; i++) {
      m[i] += (gradient[i] - m[i]) * beta1_minus;
      v[i] += (gradient[i] * gradient[i] - v[i]) * beta2_minus;
      T update = m[i] / (std::sqrt(v[i]) + epsilon);
      update += decay * var[i];
      var[i] -= lr * update;
    }
  };
  CPUKernelUtils::ParallelFor(task, lens);
}

void AdamWeightDecayCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::vector<size_t> var_shape = AnfAlgo::GetInputDeviceShape(kernel_node, VAR);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, VAR);
  gradient_dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, GRAD);
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != kAdamWeightDecayInputSize) {
    MS_LOG(EXCEPTION) << "Input number is " << input_num << ", but AdamWeightDecay needs 9 inputs.";
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != kAdamWeightDecayOutputSize) {
    MS_LOG(EXCEPTION) << "Output number is " << output_num << ", but AdamWeightDecay needs 3 outputs.";
  }
  elem_num_ = 1;
  for (size_t i : var_shape) {
    elem_num_ *= i;
  }
  if (elem_num_ < 1) {
    MS_LOG(EXCEPTION) << "Invalid parameter shape";
  }
  if (dtype_ != kNumberTypeFloat32) {
    MS_LOG(EXCEPTION) << "The dtype of parameter must be float32!";
  }
  if (gradient_dtype_ != kNumberTypeFloat32 && gradient_dtype_ != kNumberTypeFloat16) {
    MS_LOG(EXCEPTION) << "The dtype of gradient must be float32 or float16!";
  }
}

void AdamWeightDecayCPUKernel::CheckParam(const std::vector<kernel::AddressPtr> &inputs,
                                          const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.size() != kAdamWeightDecayInputSize) {
    MS_LOG(EXCEPTION) << "Input number is " << inputs.size() << ", but AdamWeightDecay needs 9 inputs.";
  }
  if (outputs.size() != kAdamWeightDecayOutputSize) {
    MS_LOG(EXCEPTION) << "Output number is " << outputs.size() << ", but AdamWeightDecay needs 3 outputs.";
  }
  size_t elem1_size = elem_num_ * kSizeFloat32;
  size_t elem2_size = gradient_dtype_ == kNumberTypeFloat16 ? elem_num_ * kSizeFloat16 : elem1_size;
  if (inputs[VAR]->size != elem1_size || inputs[M]->size != elem1_size || inputs[V]->size != elem1_size ||
      inputs[GRAD]->size != elem2_size) {
    MS_LOG(EXCEPTION) << "Error input data size!";
  }
  if (inputs[LR]->size != kSizeFloat32 || inputs[BETA1]->size != kSizeFloat32 || inputs[BETA2]->size != kSizeFloat32 ||
      inputs[EPSILON]->size != kSizeFloat32 || inputs[DECAY]->size != kSizeFloat32) {
    MS_LOG(EXCEPTION) << "The attribute beta, lr, epsilon and weight decay must be float!";
  }
}

bool AdamWeightDecayCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                      const std::vector<kernel::AddressPtr> &,
                                      const std::vector<kernel::AddressPtr> &outputs) {
  CheckParam(inputs, outputs);
  if (gradient_dtype_ == kNumberTypeFloat16) {
    LaunchFusedAdam<float, float16>(inputs, outputs);
  } else if (gradient_dtype_ == kNumberTypeFloat32) {
    LaunchAdamWeightDecay<float>(inputs, outputs);
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
