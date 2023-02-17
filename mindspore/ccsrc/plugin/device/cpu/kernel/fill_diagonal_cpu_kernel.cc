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

#include "plugin/device/cpu/kernel/fill_diagonal_cpu_kernel.h"
#include <algorithm>
#include <vector>
#include <map>
#include <memory>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "mindspore/core/ops/fill_diagonal.h"

namespace mindspore {
namespace kernel {
namespace {
const size_t kFillDiagonalInputNum = 1;
const size_t kFillDiagonalOutputNum = 1;
const size_t kInputDimIndex0 = 0;
const size_t kInputDimIndex1 = 1;
const size_t kInputMinDim = 2;
constexpr int64_t kParallelDataNums = 512 * 1024;
}  // namespace

bool FillDiagonalCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->GetPrim()->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kFillDiagonalInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kFillDiagonalOutputNum, kernel_name_);

  input_type_ = inputs[0]->GetDtype();
  auto kernel_ptr = std::dynamic_pointer_cast<ops::FillDiagonal>(base_operator);
  if (kernel_ptr == nullptr) {
    MS_LOG(ERROR) << "Init FillDiagonal kernel ptr failed.";
    return false;
  }
  fill_value_ = kernel_ptr->get_fill_value();
  wrap_ = kernel_ptr->get_wrap();
  return true;
}

int FillDiagonalCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs,
                                     const std::map<uint32_t, tensor::TensorPtr> &) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  input_shape_ = inputs[0]->GetDeviceShapeAdaptively();
  return KRET_OK;
}

bool FillDiagonalCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                      const std::vector<kernel::AddressPtr> &workspace,
                                      const std::vector<kernel::AddressPtr> &outputs) {
  if (input_type_ == kNumberTypeFloat32) {
    return LaunchKernel<float>(inputs, outputs);
  } else if (input_type_ == kNumberTypeInt32) {
    return LaunchKernel<int32_t>(inputs, outputs);
  } else if (input_type_ == kNumberTypeInt64) {
    return LaunchKernel<int64_t>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "the datatype of the input not support, support datatype: float, int32, int64.";
  }
}

template <typename T>
bool FillDiagonalCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                            const std::vector<kernel::AddressPtr> &outputs) {
  T *input_ptr = reinterpret_cast<T *>(inputs[0]->addr);
  MS_EXCEPTION_IF_NULL(input_ptr);
  T *output_ptr = reinterpret_cast<T *>(outputs[0]->addr);
  MS_EXCEPTION_IF_NULL(output_ptr);

  size_t data_nums = outputs[0]->size / sizeof(T);
  if (SizeToLong(data_nums) <= kParallelDataNums) {
    auto ret_code = memcpy_s(output_ptr, data_nums * sizeof(T), input_ptr, data_nums * sizeof(T));
    if (ret_code != EOK) {
      MS_LOG(EXCEPTION) << "Failed to copy data, memcpy_s errorno: " << ret_code;
    }
  } else {
    auto task = [this, input_ptr, output_ptr](size_t start, size_t end) {
      auto ret_code =
        memcpy_s(output_ptr + start, (end - start) * sizeof(T), input_ptr + start, (end - start) * sizeof(T));
      if (ret_code != EOK) {
        MS_LOG(EXCEPTION) << "Failed to copy data, memcpy_s errorno: " << ret_code;
      }
    };
    CPUKernelUtils::ParallelFor(task, data_nums);
  }

  int64_t height = input_shape_[kInputDimIndex0];
  int64_t width = input_shape_[kInputDimIndex1];
  int64_t size = std::min(height, width);

  int64_t stride = 0;
  for (int64_t i = (SizeToLong(input_shape_.size()) - 1); i >= 0; i--) {
    stride += static_cast<int64_t>(pow(width, i));
  }
  for (int64_t i = 0; i < size; ++i) {
    output_ptr[stride * i] = static_cast<T>(fill_value_);
  }

  if (wrap_ && input_shape_.size() == kInputMinDim && height > width + 1) {
    int64_t location = size * (size + 1);
    while (location < SizeToLong(data_nums)) {
      output_ptr[location] = static_cast<T>(fill_value_);
      location += stride;
    }
  }

  return true;
}

std::vector<KernelAttr> FillDiagonalCpuKernelMod::GetOpSupport() {
  static const std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, FillDiagonal, FillDiagonalCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
