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

#include "plugin/device/cpu/kernel/approximate_equal_cpu_kernel.h"
#include <algorithm>
#include <vector>
#include "ops/base_operator.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kApproximateEqualIutputsNum = 2;
constexpr size_t kApproximateEqualOutputsNum = 1;
constexpr size_t kXIndex = 0;
constexpr size_t kYIndex = 1;
constexpr int64_t kMaxShape = 8;
}  // namespace

void ApproximateEqualCpuKernelMod::CheckParam(const std::vector<KernelTensorPtr> &inputs,
                                              const std::vector<KernelTensorPtr> &outputs) const {
  // inputs: x, y
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kApproximateEqualIutputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kApproximateEqualOutputsNum, kernel_name_);

  auto x_shape = inputs[kXIndex]->GetShapeVector();
  auto y_shape = inputs[kYIndex]->GetShapeVector();
  if (x_shape != y_shape) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the shape of 'x' and 'y' must be the same, "
                         "but got shape of 'x': "
                      << x_shape << " and 'y': " << y_shape;
  }
  size_t x_size = x_shape.size();
  if (x_size >= kMaxShape) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the x's rank should be less than " << kMaxShape
                      << ", but got: " << x_size;
  }
  size_t y_size = y_shape.size();
  // x and y have the same size
  if (x_size != y_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the shape and dtype of 'x' and 'y' must be the same, "
                         "but got the memory size of 'x': "
                      << x_size << " and 'y': " << y_size;
  }
}

bool ApproximateEqualCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  CheckParam(inputs, outputs);
  auto kernel_ptr = std::dynamic_pointer_cast<ops::ApproximateEqual>(base_operator);
  if (kernel_ptr == nullptr) {
    MS_LOG(ERROR) << "Cast 'ApproximateEqual' ops failed!";
    return false;
  }

  tolerance_ = kernel_ptr->get_tolerance();

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  // pair = (is_match, index)
  auto pair = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!pair.first) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[pair.second].second;
  return true;
}

template <typename T>
bool ApproximateEqualCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                const std::vector<kernel::AddressPtr> &outputs) {
  const float &tolerance = this->tolerance_;
  size_t length = inputs.at(kIndex0)->size / sizeof(T);
  auto *x = reinterpret_cast<T *>(inputs.at(kIndex0)->addr);
  auto *y = reinterpret_cast<T *>(inputs.at(kIndex1)->addr);
  auto *output = reinterpret_cast<bool *>(outputs.at(kIndex0)->addr);
  auto task = [x, y, output, &tolerance](size_t start, size_t end) {
    const T tol = static_cast<T>(tolerance);
    for (size_t i = start; i < end; i++) {
      output[i] = abs(x[i] - y[i]) < tol ? true : false;
    }
  };
  ParallelLaunch(task, length, 0, this, pool_);
  return true;
}

std::vector<KernelAttr> ApproximateEqualCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ApproximateEqualFunc> &pair) { return pair.first; });
  return support_list;
}

std::vector<std::pair<KernelAttr, ApproximateEqualCpuKernelMod::ApproximateEqualFunc>>
  ApproximateEqualCpuKernelMod::func_list_{
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool),
     &ApproximateEqualCpuKernelMod::LaunchKernel<float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
     &ApproximateEqualCpuKernelMod::LaunchKernel<float>},
  };
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ApproximateEqual, ApproximateEqualCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
