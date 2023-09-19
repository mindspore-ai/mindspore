/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/dynamic_assign_cpu_kernel.h"

#include <algorithm>

#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
int DynamicAssignCpuKernelMod::Resize(const std::vector<kernel::KernelTensor *> &inputs,
                                      const std::vector<kernel::KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  input_x_dtype_ = inputs[kIndex0]->dtype_id();
  input_x_dtype_size_ = GetTypeByte(TypeIdToType(input_x_dtype_));
  return KRET_OK;
}

bool DynamicAssignCpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                                       const std::vector<kernel::KernelTensor *> &,
                                       const std::vector<kernel::KernelTensor *> &outputs) {
  if (input_x_dtype_ == kNumberTypeInt32) {
    LaunchKernel<int>(inputs, outputs);
  } else if (input_x_dtype_ == kNumberTypeInt64) {
    LaunchKernel<int64_t>(inputs, outputs);
  } else if (input_x_dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  } else if (input_x_dtype_ == kNumberTypeFloat64) {
    LaunchKernel<double>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << ", the dtype of 'input_x' must be in (int32, int64, float32, float64) on CPU, but got "
                      << TypeIdToType(input_x_dtype_)->ToString();
  }
  return true;
}

template <typename T>
void DynamicAssignCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<kernel::KernelTensor *> &) {
  const auto &input_x_shape = inputs[kIndex0]->GetShapeVector();
  const auto &input_y_shape = inputs[kIndex1]->GetShapeVector();
  if (AnfAlgo::IsShapesDynamic({input_x_shape, input_y_shape})) {
    return;
  }
  batch_size_ = 1;
  for (size_t i = 0; i < input_x_shape.size(); ++i) {
    batch_size_ *= LongToSize(input_x_shape[i]);
  }

  if (input_x_shape.size() != input_y_shape.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of 'input_x' and 'input_y' must be the same, "
                         "but got the dimension of 'input_x': "
                      << input_x_shape.size() << ", and the dimension of 'input_y': " << input_y_shape.size();
  }
  for (size_t i = 0; i < input_x_shape.size(); ++i) {
    if (input_x_shape[i] != input_y_shape[i]) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the shape of 'input_x' and 'input_y' must be the same, "
                           "but got the shape of 'input_x': "
                        << input_x_shape << ", and the shape of 'input_y': " << input_y_shape;
    }
  }
  auto *input_x = reinterpret_cast<T *>(inputs[0]->device_ptr());
  auto *input_y = reinterpret_cast<T *>(inputs[1]->device_ptr());
  auto max_size = inputs[0]->size();
  size_t total_size = input_x_dtype_size_ * batch_size_;
  if (total_size > max_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', memcpy size must be less than or equal to max size, but got memcpy size: " << total_size
                      << " and max size: " << max_size;
  }
  int ret = memcpy_s(input_x, total_size, input_y, total_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memcpy_s error.  Error no: " << ret;
  }
}

void DynamicAssignCpuKernelMod::UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                                         const std::vector<KernelTensor *> &outputs) {
  const auto &input_shape = inputs[kIndex0]->GetShapeVector();
  outputs[kIndex0]->SetShapeVector(input_shape);
  outputs[kIndex0]->set_size(input_x_dtype_size_ * batch_size_);
}

std::vector<KernelAttr> DynamicAssignCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64)};

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, DynamicAssign, DynamicAssignCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
