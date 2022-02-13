/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/cpu/kernel/fused_cast_adam_weight_decay_cpu_kernel.h"
#include <cmath>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "nnacl/fp32/adam_fp32.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
static constexpr size_t BATCH_SIZE = 10000;
static constexpr float MIN_GLOBAL_NORM = 1e-10;
void FusedCastAdamWeightDecayCpuKernelMod::LaunchFusedCastAdamFp32(const std::vector<AddressPtr> &inputs,
                                                                   const std::vector<AddressPtr> &) {
  auto m = reinterpret_cast<float *>(inputs[M]->addr);
  auto v = reinterpret_cast<float *>(inputs[V]->addr);
  auto lr = reinterpret_cast<float *>(inputs[LR]->addr)[kScalarIndex];
  auto beta1 = reinterpret_cast<float *>(inputs[BETA1]->addr)[kScalarIndex];
  auto beta2 = reinterpret_cast<float *>(inputs[BETA2]->addr)[kScalarIndex];
  auto epsilon = reinterpret_cast<float *>(inputs[EPSILON]->addr)[kScalarIndex];
  auto decay = reinterpret_cast<float *>(inputs[DECAY]->addr)[kScalarIndex];
  auto gradient16 = reinterpret_cast<float16 *>(inputs[GRAD]->addr);
  auto var = reinterpret_cast<float *>(inputs[VAR]->addr);
  auto global_norm = reinterpret_cast<float *>(inputs[GLOBAL_NORM]->addr)[kScalarIndex];
  if (global_norm < MIN_GLOBAL_NORM) {
    global_norm = 1.0f;
  }
  auto global_norm_reciprocal = 1.0f / global_norm;
  const auto beta1_minus = 1 - beta1;
  const auto beta2_minus = 1 - beta2;

  // multithreading
  size_t lens = inputs[VAR]->size > 0 ? static_cast<size_t>(inputs[VAR]->size / kSizeFloat32) : 1;
  std::function<void(size_t, size_t)> task;

  task = [&](size_t start, size_t end) {
    size_t i = FusedCastAdamFp32(var, m, v, lr, beta1, beta2, epsilon, decay, reinterpret_cast<int16_t *>(gradient16),
                                 global_norm_reciprocal, start, end);
    // remaining
    for (; i < end; i++) {
      auto temp = static_cast<float>(gradient16[i]) * global_norm_reciprocal;
      m[i] += (temp - m[i]) * beta1_minus;
      v[i] += (temp * temp - v[i]) * beta2_minus;
      auto update = m[i] / (std::sqrt(v[i]) + epsilon);
      update += decay * var[i];
      var[i] -= lr * update;
    }
  };
  CPUKernelUtils::ParallelFor(task, lens, BATCH_SIZE);
}

void FusedCastAdamWeightDecayCpuKernelMod::LaunchFusedCastAdamFp16(const std::vector<AddressPtr> &inputs,
                                                                   const std::vector<AddressPtr> &) {
  auto m = reinterpret_cast<float *>(inputs[M]->addr);
  auto v = reinterpret_cast<float *>(inputs[V]->addr);
  auto lr = reinterpret_cast<float *>(inputs[LR]->addr)[kScalarIndex];
  auto beta1 = reinterpret_cast<float *>(inputs[BETA1]->addr)[kScalarIndex];
  auto beta2 = reinterpret_cast<float *>(inputs[BETA2]->addr)[kScalarIndex];
  auto epsilon = reinterpret_cast<float *>(inputs[EPSILON]->addr)[kScalarIndex];
  auto decay = reinterpret_cast<float *>(inputs[DECAY]->addr)[kScalarIndex];
  auto gradient16 = reinterpret_cast<float16 *>(inputs[GRAD]->addr);
  auto var16 = reinterpret_cast<float16 *>(inputs[VAR]->addr);
  auto global_norm = reinterpret_cast<float *>(inputs[GLOBAL_NORM]->addr)[kScalarIndex];
  if (global_norm < MIN_GLOBAL_NORM) {
    global_norm = 1.0f;
  }
  auto global_norm_reciprocal = 1.0f / global_norm;
  const auto beta1_minus = 1 - beta1;
  const auto beta2_minus = 1 - beta2;

  // multithreading
  size_t lens = inputs[VAR]->size > 0 ? static_cast<size_t>(inputs[VAR]->size / kSizeFloat16) : 1;
  std::function<void(size_t, size_t)> task;

  task = [&](size_t start, size_t end) {
    size_t i = FusedCastAdamFp16(reinterpret_cast<int16_t *>(var16), m, v, lr, beta1, beta2, epsilon, decay,
                                 reinterpret_cast<int16_t *>(gradient16), global_norm_reciprocal, start, end);
    // remaining
    for (; i < end; i++) {
      auto temp_var = static_cast<float>(var16[i]);
      auto temp_grad = static_cast<float>(gradient16[i]) * global_norm_reciprocal;
      m[i] += (temp_grad - m[i]) * beta1_minus;
      v[i] += (temp_grad * temp_grad - v[i]) * beta2_minus;
      auto update = m[i] / (std::sqrt(v[i]) + epsilon);
      update += decay * temp_var;
      temp_var -= lr * update;
      var16[i] = static_cast<float16>(temp_var);
    }
  };
  CPUKernelUtils::ParallelFor(task, lens, BATCH_SIZE);
}

void FusedCastAdamWeightDecayCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  std::vector<size_t> var_shape = AnfAlgo::GetInputDeviceShape(kernel_node, VAR);
  var_dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, VAR);
  gradient_dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, GRAD);
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != kFusedCastAdamWeightDecayInputNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be "
                      << kFusedCastAdamWeightDecayInputNum << ", but got: " << input_num;
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != kFusedCastAdamWeightDecayOutputNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs should be "
                      << kFusedCastAdamWeightDecayOutputNum << ", but got: " << output_num;
  }
  elem_num_ = 1;
  for (size_t i : var_shape) {
    elem_num_ *= i;
  }
  if (elem_num_ < 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of 'var' should not be zero.";
  }
  if (gradient_dtype_ != kNumberTypeFloat16) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of 'gradient' should be float16, but got "
                      << TypeIdToType(gradient_dtype_)->ToString();
  }
  if (var_dtype_ != kNumberTypeFloat32 && var_dtype_ != kNumberTypeFloat16) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of 'var' should be float16 or float32, but got "
                      << TypeIdToType(var_dtype_)->ToString();
  }
}

void FusedCastAdamWeightDecayCpuKernelMod::CheckParam(const std::vector<kernel::AddressPtr> &inputs,
                                                      const std::vector<kernel::AddressPtr> &outputs) const {
  if (inputs.size() != kFusedCastAdamWeightDecayInputNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be "
                      << kFusedCastAdamWeightDecayInputNum << ", but got: " << inputs.size();
  }
  if (outputs.size() != kFusedCastAdamWeightDecayOutputNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs should be "
                      << kFusedCastAdamWeightDecayOutputNum << ", but got: " << outputs.size();
  }
  size_t elem_size_fp32 = elem_num_ * kSizeFloat32;
  size_t elem_size_fp16 = elem_num_ * kSizeFloat16;
  size_t var_size = var_dtype_ == kNumberTypeFloat16 ? elem_size_fp16 : elem_size_fp32;
  if (inputs[VAR]->size != var_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'var' should be " << var_size
                      << ", but got " << inputs[VAR]->size;
  }
  if (inputs[M]->size != elem_size_fp32) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'm' should be " << elem_size_fp32
                      << ", but got " << inputs[M]->size;
  }
  if (inputs[V]->size != elem_size_fp32) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'v' should be " << elem_size_fp32
                      << ", but got " << inputs[V]->size;
  }
  if (inputs[GRAD]->size != elem_size_fp16) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'gradient' should be " << elem_size_fp16
                      << ", but got " << inputs[GRAD]->size;
  }
  if (inputs[LR]->size != kSizeFloat32) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the type of 'lr' should be float32, but got 'lr': " << inputs[LR];
  }
  if (inputs[BETA1]->size != kSizeFloat32) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the type of 'beta1' should be float32, but got 'beta1': " << inputs[BETA1];
  }
  if (inputs[BETA2]->size != kSizeFloat32) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the type of 'beta2' should be float32, but got 'beta2': " << inputs[BETA2];
  }
  if (inputs[EPSILON]->size != kSizeFloat32) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the type of 'epsilon' should be float32, but got 'epsilon': " << inputs[EPSILON];
  }
  if (inputs[DECAY]->size != kSizeFloat32) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the type of 'decay' should be float32, but got 'decay': " << inputs[DECAY];
  }
}

bool FusedCastAdamWeightDecayCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                                  const std::vector<kernel::AddressPtr> &,
                                                  const std::vector<kernel::AddressPtr> &outputs) {
  CheckParam(inputs, outputs);
  if (var_dtype_ == kNumberTypeFloat16) {
    LaunchFusedCastAdamFp16(inputs, outputs);
  } else {
    LaunchFusedCastAdamFp32(inputs, outputs);
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
