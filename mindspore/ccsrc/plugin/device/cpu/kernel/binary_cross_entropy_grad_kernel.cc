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

#include "plugin/device/cpu/kernel/binary_cross_entropy_grad_kernel.h"
#include <map>

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kBceGradInputsNumWithWeight = 5;
constexpr size_t kBceGradOutputsNum = 1;
}  // namespace

template <typename T>
void BinaryCrossEntropyGradCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                      const std::vector<KernelTensor *> &outputs) {
  const auto *input_x = reinterpret_cast<T *>(inputs[0]->device_ptr());
  const auto *input_y = reinterpret_cast<T *>(inputs[1]->device_ptr());
  const auto *dloss = reinterpret_cast<T *>(inputs[2]->device_ptr());
  const T *weight = reinterpret_cast<T *>(inputs[3]->device_ptr());
  auto *dx = reinterpret_cast<T *>(outputs[0]->device_ptr());
  auto epsilon = static_cast<T>(1e-12);
  auto one = static_cast<T>(1);

  std::function<void(size_t, size_t)> func;
  if (reduction_ == Reduction::NONE) {
    if (weight != nullptr) {
      func = [&](size_t start, size_t end) -> void {
        for (size_t i = start; i < end; i++) {
          T denominator = ((input_x[i] * (one - input_x[i])) > epsilon) ? (input_x[i] * (one - input_x[i])) : epsilon;
          T value = weight[i] * (input_x[i] - input_y[i]) / denominator;
          dx[i] = value * dloss[i];
        }
      };
    } else {
      func = [&](size_t start, size_t end) -> void {
        for (size_t i = start; i < end; i++) {
          T denominator = ((input_x[i] * (one - input_x[i])) > epsilon) ? (input_x[i] * (one - input_x[i])) : epsilon;
          T value = (input_x[i] - input_y[i]) / denominator;
          dx[i] = value * dloss[i];
        }
      };
    }
  } else {
    T dloss1 = dloss[0];
    if (reduction_ == Reduction::MEAN) {
      dloss1 = dloss[0] / static_cast<T>(input_size_);
    }
    if (weight != nullptr) {
      func = [&](size_t start, size_t end) -> void {
        for (size_t i = start; i < end; i++) {
          T denominator = ((input_x[i] * (one - input_x[i])) > epsilon) ? (input_x[i] * (one - input_x[i])) : epsilon;
          T value = weight[i] * (input_x[i] - input_y[i]) / denominator;
          dx[i] = value * dloss1;
        }
      };
    } else {
      func = [&](size_t start, size_t end) -> void {
        for (size_t i = start; i < end; i++) {
          T denominator = ((input_x[i] * (one - input_x[i])) > epsilon) ? (input_x[i] * (one - input_x[i])) : epsilon;
          T value = (input_x[i] - input_y[i]) / denominator;
          dx[i] = value * dloss1;
        }
      };
    }
  }
  ParallelLaunchAutoSearch(func, input_size_, this, &parallel_search_info_);
}

bool BinaryCrossEntropyGradCpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                                const std::vector<KernelTensor *> &,
                                                const std::vector<KernelTensor *> &outputs) {
  const size_t expect_inputs_num = kBceGradInputsNumWithWeight;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), expect_inputs_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kBceGradOutputsNum, kernel_name_);
  auto reduction = inputs[kIndex4]->GetValueWithCheck<int64_t>();
  reduction_ = static_cast<Reduction>(reduction);
  if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of input must be float16 or float32, but got "
                      << TypeIdToType(dtype_)->ToString();
  }
  return true;
}

bool BinaryCrossEntropyGradCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &outputs) {
  dtype_ = inputs[kIndex0]->dtype_id();
  return true;
}

int BinaryCrossEntropyGradCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != 0) {
    return ret;
  }
  auto input_shape = inputs[kIndex0]->GetShapeVector();
  input_size_ = 1;
  for (size_t i = 0; i < input_shape.size(); i++) {
    input_size_ *= LongToSize(input_shape[i]);
  }
  return KRET_OK;
}

std::vector<KernelAttr> BinaryCrossEntropyGradCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> kernel_attr_list = {KernelAttr()
                                                       .AddInputAttr(kNumberTypeFloat16)
                                                       .AddInputAttr(kNumberTypeFloat16)
                                                       .AddInputAttr(kNumberTypeFloat16)
                                                       .AddOptionalInputAttr(kNumberTypeFloat16)
                                                       .AddInputAttr(kNumberTypeInt64)
                                                       .AddOutputAttr(kNumberTypeFloat16),
                                                     KernelAttr()
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddOptionalInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeInt64)
                                                       .AddOutputAttr(kNumberTypeFloat32)};

  return kernel_attr_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, BinaryCrossEntropyGrad, BinaryCrossEntropyGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
