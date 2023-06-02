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

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSizeFloat32 = sizeof(float);
constexpr size_t kSizeFloat16 = sizeof(float16);
constexpr size_t kScalarIndex = 0;
constexpr size_t kFusedCastAdamWeightDecayInputNum = 10;
constexpr size_t kFusedCastAdamWeightDecayOutputNum = 3;
constexpr size_t kBatchSize = 10000;
constexpr float kMinGlobalNorm = 1e-10;
constexpr size_t kVarIndex = 0;
constexpr size_t kMIndex = 1;
constexpr size_t kVIndex = 2;
constexpr size_t kLRIndex = 3;
constexpr size_t kBeta1Index = 4;
constexpr size_t kBeta2Index = 5;
constexpr size_t kEpsIndex = 6;
constexpr size_t kDecayIndex = 7;
constexpr size_t kGradIndex = 8;
constexpr size_t kGlobalNormIndex = 9;
}  // namespace

void FusedCastAdamWeightDecayCpuKernelMod::LaunchFusedCastAdamFp32(const std::vector<AddressPtr> &inputs,
                                                                   const std::vector<AddressPtr> &) const {
  auto m = reinterpret_cast<float *>(inputs[kMIndex]->addr);
  auto v = reinterpret_cast<float *>(inputs[kVIndex]->addr);
  auto lr = reinterpret_cast<float *>(inputs[kLRIndex]->addr)[kScalarIndex];
  auto beta1 = reinterpret_cast<float *>(inputs[kBeta1Index]->addr)[kScalarIndex];
  auto beta2 = reinterpret_cast<float *>(inputs[kBeta2Index]->addr)[kScalarIndex];
  auto epsilon = reinterpret_cast<float *>(inputs[kEpsIndex]->addr)[kScalarIndex];
  auto decay = reinterpret_cast<float *>(inputs[kDecayIndex]->addr)[kScalarIndex];
  auto var = reinterpret_cast<float *>(inputs[kVarIndex]->addr);
  auto global_norm = reinterpret_cast<float *>(inputs[kGlobalNormIndex]->addr)[kScalarIndex];
  if (global_norm < kMinGlobalNorm) {
    global_norm = 1.0f;
  }
  auto global_norm_reciprocal = 1.0f / global_norm;
  const auto beta1_minus = 1 - beta1;
  const auto beta2_minus = 1 - beta2;

  // multithreading
  size_t lens = inputs[kVarIndex]->size > 0 ? static_cast<size_t>(inputs[kVarIndex]->size / kSizeFloat32) : 1;
  std::function<void(size_t, size_t)> task;

  if (gradient_dtype_ == kNumberTypeFloat16) {
    float16 *gradient16 = reinterpret_cast<float16 *>(inputs[kGradIndex]->addr);
    task = [&](size_t start, size_t end) {
      size_t i = FusedCastAdamFp32Fp16(var, reinterpret_cast<int16_t *>(gradient16), m, v, lr, beta1, beta2, epsilon,
                                       decay, global_norm_reciprocal, start, end);
      for (; i < end; ++i) {
        auto temp = static_cast<float>(gradient16[i]) * global_norm_reciprocal;
        m[i] += (temp - m[i]) * beta1_minus;
        v[i] += (temp * temp - v[i]) * beta2_minus;
        auto update = m[i] / (std::sqrt(v[i]) + epsilon);
        update += decay * var[i];
        var[i] -= lr * update;
      }
    };
  } else {
    float *gradient32 = reinterpret_cast<float *>(inputs[kGradIndex]->addr);
    task = [&](size_t start, size_t end) {
      size_t i = FusedCastAdamFp32Fp32(var, gradient32, m, v, lr, beta1, beta2, epsilon, decay, global_norm_reciprocal,
                                       start, end);
      for (; i < end; ++i) {
        auto temp = gradient32[i] * global_norm_reciprocal;
        m[i] += (temp - m[i]) * beta1_minus;
        v[i] += (temp * temp - v[i]) * beta2_minus;
        auto update = m[i] / (std::sqrt(v[i]) + epsilon);
        update += decay * var[i];
        var[i] -= lr * update;
      }
    };
  }

  CPUKernelUtils::ParallelFor(task, lens, kBatchSize);
}

void FusedCastAdamWeightDecayCpuKernelMod::LaunchFusedCastAdamFp16(const std::vector<AddressPtr> &inputs,
                                                                   const std::vector<AddressPtr> &) const {
  auto m = reinterpret_cast<float *>(inputs[kMIndex]->addr);
  auto v = reinterpret_cast<float *>(inputs[kVIndex]->addr);
  auto lr = reinterpret_cast<float *>(inputs[kLRIndex]->addr)[kScalarIndex];
  auto beta1 = reinterpret_cast<float *>(inputs[kBeta1Index]->addr)[kScalarIndex];
  auto beta2 = reinterpret_cast<float *>(inputs[kBeta2Index]->addr)[kScalarIndex];
  auto epsilon = reinterpret_cast<float *>(inputs[kEpsIndex]->addr)[kScalarIndex];
  auto decay = reinterpret_cast<float *>(inputs[kDecayIndex]->addr)[kScalarIndex];
  auto var16 = reinterpret_cast<float16 *>(inputs[kVarIndex]->addr);
  auto global_norm = reinterpret_cast<float *>(inputs[kGlobalNormIndex]->addr)[kScalarIndex];
  if (global_norm < kMinGlobalNorm) {
    global_norm = 1.0f;
  }

  auto global_norm_reciprocal = 1.0f / global_norm;
  const auto beta1_minus = 1 - beta1;
  const auto beta2_minus = 1 - beta2;

  // multithreading
  size_t lens = inputs[kVarIndex]->size > 0 ? static_cast<size_t>(inputs[kVarIndex]->size / kSizeFloat16) : 1;
  std::function<void(size_t, size_t)> task;

  if (gradient_dtype_ == kNumberTypeFloat16) {
    float16 *gradient16 = reinterpret_cast<float16 *>(inputs[kGradIndex]->addr);
    task = [&](size_t start, size_t end) {
      size_t i = FusedCastAdamFp16Fp16(reinterpret_cast<int16_t *>(var16), reinterpret_cast<int16_t *>(gradient16), m,
                                       v, lr, beta1, beta2, epsilon, decay, global_norm_reciprocal, start, end);
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
  } else {
    float *gradient32 = reinterpret_cast<float *>(inputs[kGradIndex]->addr);
    task = [&](size_t start, size_t end) {
      size_t i = FusedCastAdamFp16Fp32(reinterpret_cast<int16_t *>(var16), gradient32, m, v, lr, beta1, beta2, epsilon,
                                       decay, global_norm_reciprocal, start, end);
      for (; i < end; i++) {
        auto temp_var = static_cast<float>(var16[i]);
        auto temp_grad = gradient32[i] * global_norm_reciprocal;
        m[i] += (temp_grad - m[i]) * beta1_minus;
        v[i] += (temp_grad * temp_grad - v[i]) * beta2_minus;
        auto update = m[i] / (std::sqrt(v[i]) + epsilon);
        update += decay * temp_var;
        temp_var -= lr * update;
        var16[i] = static_cast<float16>(temp_var);
      }
    };
  }

  CPUKernelUtils::ParallelFor(task, lens, kBatchSize);
}

void FusedCastAdamWeightDecayCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  auto var_shape = AnfAlgo::GetInputDeviceShape(kernel_node, kVarIndex);
  if (AnfAlgo::IsShapesDynamic({var_shape})) {
    return;
  }
  var_dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, kVarIndex);
  gradient_dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, kGradIndex);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != kFusedCastAdamWeightDecayInputNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be "
                      << kFusedCastAdamWeightDecayInputNum << ", but got: " << input_num;
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != kFusedCastAdamWeightDecayOutputNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs must be "
                      << kFusedCastAdamWeightDecayOutputNum << ", but got: " << output_num;
  }
  elem_num_ = 1;
  for (auto i : var_shape) {
    elem_num_ *= static_cast<size_t>(i);
  }
  if (elem_num_ < 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of 'var' can not be zero.";
  }

  if (gradient_dtype_ != kNumberTypeFloat32 && gradient_dtype_ != kNumberTypeFloat16) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of 'gradient' must be float16 or float32, but got "
                      << TypeIdToType(gradient_dtype_)->ToString();
  }
  if (var_dtype_ != kNumberTypeFloat32 && var_dtype_ != kNumberTypeFloat16) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of 'var' must be float16 or float32, but got "
                      << TypeIdToType(var_dtype_)->ToString();
  }
}

void FusedCastAdamWeightDecayCpuKernelMod::CheckParam(const std::vector<kernel::AddressPtr> &inputs,
                                                      const std::vector<kernel::AddressPtr> &outputs) const {
  if (inputs.size() != kFusedCastAdamWeightDecayInputNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be "
                      << kFusedCastAdamWeightDecayInputNum << ", but got: " << inputs.size();
  }
  if (outputs.size() != kFusedCastAdamWeightDecayOutputNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs must be "
                      << kFusedCastAdamWeightDecayOutputNum << ", but got: " << outputs.size();
  }
  size_t elem_size_fp32 = elem_num_ * kSizeFloat32;
  size_t elem_size_fp16 = elem_num_ * kSizeFloat16;
  size_t var_size = var_dtype_ == kNumberTypeFloat16 ? elem_size_fp16 : elem_size_fp32;
  size_t grad_size = gradient_dtype_ == kNumberTypeFloat16 ? elem_size_fp16 : elem_size_fp32;
  if (inputs[kVarIndex]->size != var_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'var' must be " << var_size << ", but got "
                      << inputs[kVarIndex]->size;
  }
  if (inputs[kMIndex]->size != elem_size_fp32) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'm' must be " << elem_size_fp32
                      << ", but got " << inputs[kMIndex]->size;
  }
  if (inputs[kVIndex]->size != elem_size_fp32) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'v' must be " << elem_size_fp32
                      << ", but got " << inputs[kVIndex]->size;
  }
  if (inputs[kGradIndex]->size != grad_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'gradient' must be " << grad_size
                      << ", but got " << inputs[kGradIndex]->size;
  }
  if (inputs[kLRIndex]->size != kSizeFloat32) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the type of 'lr' must be float32, but got 'lr': " << inputs[kLRIndex];
  }
  if (inputs[kBeta1Index]->size != kSizeFloat32) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the type of 'beta1' must be float32, but got 'beta1': " << inputs[kBeta1Index];
  }
  if (inputs[kBeta2Index]->size != kSizeFloat32) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the type of 'beta2' must be float32, but got 'beta2': " << inputs[kBeta2Index];
  }
  if (inputs[kEpsIndex]->size != kSizeFloat32) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the type of 'epsilon' must be float32, but got 'epsilon': " << inputs[kEpsIndex];
  }
  if (inputs[kDecayIndex]->size != kSizeFloat32) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the type of 'decay' must be float32, but got 'decay': " << inputs[kDecayIndex];
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

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, FusedCastAdamWeightDecay, FusedCastAdamWeightDecayCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
