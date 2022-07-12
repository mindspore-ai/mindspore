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
constexpr size_t kGluGradInputsNum = 2;
constexpr size_t kGluGradOutputsNum = 1;
const int64_t kEvenNum = 2;
}  // namespace

void GluGradCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  auto axis_value = common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "axis");
  grad_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  x_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 1);

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
                      << axis_value << ", x.shape=" << Vector2Str(x_shape_)
                      << " and grad.shape=" << Vector2Str(grad_shape_) << ".";
  }
}

bool GluGradCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                                 const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kGluGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kGluGradOutputsNum, kernel_name_);
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
void GluGradCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  const auto *input0 = reinterpret_cast<T *>(inputs[0]->addr);
  const auto *input1 = reinterpret_cast<T *>(inputs[1]->addr);
  auto *output = reinterpret_cast<T *>(outputs[0]->addr);
  std::vector<int64_t> shape = x_shape_;
  int64_t dim = axis_;
  size_t lens = outputs[0]->size > 0 ? outputs[0]->size / sizeof(T) : 1;
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
