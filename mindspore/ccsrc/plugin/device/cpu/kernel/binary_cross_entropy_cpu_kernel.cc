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

#include "plugin/device/cpu/kernel/binary_cross_entropy_cpu_kernel.h"
#include <map>
#include "mindspore/core/ops/binary_cross_entropy.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kBceInputsNumWithWeight = 3;
constexpr size_t kBceOutputsNum = 1;
}  // namespace

template <typename T>
void BinaryCrossEntropyCpuKernelMod::LaunchToScalar(const int &input_size, const ReductionType &reduction, T *loss,
                                                    T *tmp_loss) const {
  if (input_size % 2 == 1) {
    tmp_loss[0] += tmp_loss[input_size - 1];
  }

  for (int stride = input_size / 2; stride > 0; stride = stride / 2) {
    for (int i = 0; i < stride; i++) {
      tmp_loss[i] += tmp_loss[i + stride];
    }
    if (stride > 2 && stride % 2 == 1) {
      tmp_loss[0] += tmp_loss[stride - 1];
    }
  }

  loss[0] = tmp_loss[0];
  if (reduction == kMean) {
    loss[0] /= static_cast<T>(input_size);
  }
}

template <typename T>
inline void CheckInput(T x) {
  auto zero = static_cast<T>(0);
  auto one = static_cast<T>(1);
  if (x > one || x < zero) {
    MS_LOG(EXCEPTION) << "For 'BinaryCrossEntropy', the value of 'input_x' must be between 0 and 1, but got value: "
                      << x;
  }
}

template <typename T>
void BinaryCrossEntropyCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                  const std::vector<AddressPtr> &outputs) {
  const auto *input_x = reinterpret_cast<T *>(inputs[0]->addr);
  const auto *input_y = reinterpret_cast<T *>(inputs[1]->addr);
  const T *weight = weight_defined_ ? reinterpret_cast<T *>(inputs[2]->addr) : nullptr;
  auto *loss = reinterpret_cast<T *>(outputs[0]->addr);
  std::vector<T> tmp_loss(input_size_);
  auto epsilon = static_cast<T>(1e-12);
  auto one = static_cast<T>(1);

  std::function<void(size_t, size_t)> func;
  if (reduction_ == kNone) {
    if (weight_defined_) {
      func = [&](size_t start, size_t end) -> void {
        for (size_t i = start; i < end; i++) {
          CheckInput(input_x[i]);
          auto value = static_cast<T>(-weight[i] * (input_y[i] * log(input_x[i] + epsilon) +
                                                    (one - input_y[i]) * log(one - input_x[i] + epsilon)));
          loss[i] = value;
        }
      };
    } else {
      func = [&](size_t start, size_t end) -> void {
        for (size_t i = start; i < end; i++) {
          CheckInput(input_x[i]);
          auto value = static_cast<T>(
            -(input_y[i] * log(input_x[i] + epsilon) + (one - input_y[i]) * log(one - input_x[i] + epsilon)));
          loss[i] = value;
        }
      };
    }
  } else {
    if (weight_defined_) {
      func = [&](size_t start, size_t end) -> void {
        for (size_t i = start; i < end; i++) {
          CheckInput(input_x[i]);
          auto value = static_cast<T>(-weight[i] * (input_y[i] * log(input_x[i] + epsilon) +
                                                    (one - input_y[i]) * log(one - input_x[i] + epsilon)));
          tmp_loss[i] = value;
        }
      };
    } else {
      func = [&](size_t start, size_t end) -> void {
        for (size_t i = start; i < end; i++) {
          CheckInput(input_x[i]);
          auto value = static_cast<T>(
            -(input_y[i] * log(input_x[i] + epsilon) + (one - input_y[i]) * log(one - input_x[i] + epsilon)));
          tmp_loss[i] = value;
        }
      };
    }
  }
  ParallelLaunchAutoSearch(func, input_size_, this, &parallel_search_info_);

  if (reduction_ != kNone) {
    LaunchToScalar<T>(input_size_, reduction_, loss, tmp_loss.data());
  }
}

bool BinaryCrossEntropyCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                            const std::vector<AddressPtr> &outputs) {
  const size_t expect_inputs_num = weight_defined_ ? kBceInputsNumWithWeight : kBceInputsNumWithWeight - 1;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), expect_inputs_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kBceOutputsNum, kernel_name_);
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

bool BinaryCrossEntropyCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::BinaryCrossEntropy>(base_operator);
  if (kernel_ptr == nullptr) {
    MS_LOG(ERROR) << "cast BinaryCrossEntropy ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();
  size_t input_num = inputs.size();
  weight_defined_ = (input_num == kBceInputsNumWithWeight);
  dtype_ = inputs[kIndex0]->GetDtype();

  const auto reduction = kernel_ptr->get_reduction();
  if (reduction == Reduction::NONE) {
    reduction_ = kNone;
  } else if (reduction == Reduction::MEAN) {
    reduction_ = kMean;
  } else {
    reduction_ = kSum;
  }
  return true;
}

int BinaryCrossEntropyCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                           const std::vector<KernelTensorPtr> &inputs,
                                           const std::vector<KernelTensorPtr> &outputs,
                                           const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs);
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

std::vector<KernelAttr> BinaryCrossEntropyCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> kernel_attr_list = {
    KernelAttr()
      .AddInputAttr(kNumberTypeFloat16)
      .AddInputAttr(kNumberTypeFloat16)
      .AddInputAttr(kNumberTypeFloat16)
      .AddOutputAttr(kNumberTypeFloat16),
    KernelAttr()
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeFloat32)
      .AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)};

  return kernel_attr_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, BinaryCrossEntropy, BinaryCrossEntropyCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
