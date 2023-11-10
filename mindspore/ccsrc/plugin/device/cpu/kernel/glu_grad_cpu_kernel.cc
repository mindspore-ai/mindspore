/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/cpu/kernel/glu_grad_cpu_kernel.h"
#include <functional>
#include <vector>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
const int64_t kEvenNum = 2;
}  // namespace

int GluGradCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  dtype_ = inputs[kIndex0]->dtype_id();
  auto axis_value = GetValue<int64_t>(primitive_->GetAttr("axis"));
  grad_shape_ = inputs[kIndex0]->GetShapeVector();
  x_shape_ = inputs[kIndex1]->GetShapeVector();

  int64_t rank = SizeToLong(x_shape_.size());
  if (axis_value < -rank || axis_value >= rank) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'axis' must be in range [" << -rank << ", " << rank
                      << "), but got " << axis_value << ".";
  }
  if (axis_value < 0) {
    axis_ = axis_value + rank;
  } else {
    axis_ = axis_value;
  }

  if (x_shape_[axis_] % kEvenNum != 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', x.shape[" << axis_value << "] must be even, but got "
                      << x_shape_[axis_] << ".";
  }

  auto expected_grad_shape = x_shape_;
  expected_grad_shape[axis_] /= kEvenNum;
  if (grad_shape_ != expected_grad_shape) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', x.shape must be euqal to grad.shape except for grad.shape[axis]=x.shape[axis]"
                         "/2,  but got axis="
                      << axis_value << ", x.shape=" << x_shape_ << " and grad.shape=" << grad_shape_ << ".";
  }
  return KRET_OK;
}

bool GluGradCpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                                 const std::vector<kernel::KernelTensor *> &,
                                 const std::vector<kernel::KernelTensor *> &outputs) {
  if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat64) {
    LaunchKernel<double>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of x must be float16, float32 or float64, but got "
                      << TypeIdLabel(dtype_) << ".";
  }
  return true;
}

template <typename T>
void GluGradCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  const auto *input0 = static_cast<T *>(inputs[0]->device_ptr());
  const auto *input1 = static_cast<T *>(inputs[1]->device_ptr());
  auto *output = static_cast<T *>(outputs[0]->device_ptr());
  std::vector<int64_t> shape = x_shape_;
  int64_t dim = axis_;
  size_t lens = outputs[0]->size() > 0 ? outputs[0]->size() / sizeof(T) : 1;
  auto task = [&input0, &input1, &output, &shape, &dim](const size_t start, const size_t end) {
    int64_t input_num = std::accumulate(shape.cbegin(), shape.cend(), 1, std::multiplies<int64_t>());
    int num = input_num;
    for (int64_t m = 0; m <= dim; m++) {
      if (m < dim) {
        num = num / shape[m];
      } else if (m == dim) {
        num = num / 2;
      }
    }
    int64_t n_m = 1;
    int64_t size_m = 0;
    int64_t grad_offset_b = 0;
    int64_t grad_offset_a = 0;
    for (int i = 0; i < input_num; i++) {
      if (n_m % 2 != 0) {
        *(output + i) = (T(1.0) / (T(1.0) + exp(-(*(input1 + (i + num)))))) * (*(input0 + grad_offset_b));
        grad_offset_b += 1;
        size_m = size_m + 1;
        if (size_m == num) {
          n_m += 1;
          size_m = 0;
        }
      } else {
        *(output + i) = *(input1 + (i - num)) * (T(1.0) / (T(1.0) + exp(-(*(input1 + i))))) *
                        (T(1.0) - (T(1.0) / (T(1.0) + exp(-(*(input1 + i)))))) * (*(input0 + grad_offset_a));
        grad_offset_a += 1;
        size_m = size_m + 1;
        if (size_m == num) {
          n_m += 1;
          size_m = 0;
        }
      }
    }
  };
  ParallelLaunchAutoSearch(task, lens, this, &parallel_search_info_);
}

std::vector<KernelAttr> GluGradCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, GluGrad, GluGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
