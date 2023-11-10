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

bool FillDiagonalCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kFillDiagonalInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kFillDiagonalOutputNum, kernel_name_);

  input_type_ = inputs[0]->dtype_id();
  fill_value_ = GetValue<float>(primitive_->GetAttr(ops::kFillValue));
  wrap_ = GetValue<bool>(primitive_->GetAttr(ops::kWrap));

  if (IsOneOfUnsignedType(input_type_) && fill_value_ < 0) {
    MS_LOG(ERROR) << "For " << kernel_name_ << ", [file_value] should be non_negative for input of unsigned type.";
    return false;
  }
  return true;
}

int FillDiagonalCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  auto ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  input_shape_ = inputs[0]->GetDeviceShapeVector();
  return KRET_OK;
}

bool FillDiagonalCpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                                      const std::vector<kernel::KernelTensor *> &workspace,
                                      const std::vector<kernel::KernelTensor *> &outputs) {
  if (input_type_ == kNumberTypeFloat16) {
    return LaunchKernel<float16>(inputs, outputs);
  } else if (input_type_ == kNumberTypeFloat32) {
    return LaunchKernel<float>(inputs, outputs);
  } else if (input_type_ == kNumberTypeFloat64) {
    return LaunchKernel<double>(inputs, outputs);
  } else if (input_type_ == kNumberTypeUInt8) {
    return LaunchKernel<uint8_t>(inputs, outputs);
  } else if (input_type_ == kNumberTypeUInt16) {
    return LaunchKernel<uint16_t>(inputs, outputs);
  } else if (input_type_ == kNumberTypeUInt32) {
    return LaunchKernel<uint32_t>(inputs, outputs);
  } else if (input_type_ == kNumberTypeUInt64) {
    return LaunchKernel<uint64_t>(inputs, outputs);
  } else if (input_type_ == kNumberTypeInt8) {
    return LaunchKernel<int8_t>(inputs, outputs);
  } else if (input_type_ == kNumberTypeInt16) {
    return LaunchKernel<int16_t>(inputs, outputs);
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
bool FillDiagonalCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                            const std::vector<kernel::KernelTensor *> &outputs) {
  T *input_ptr = reinterpret_cast<T *>(inputs[0]->device_ptr());
  MS_EXCEPTION_IF_NULL(input_ptr);
  T *output_ptr = reinterpret_cast<T *>(outputs[0]->device_ptr());
  MS_EXCEPTION_IF_NULL(output_ptr);

  size_t data_nums = outputs[0]->size() / sizeof(T);
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
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt8),
    KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
    KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, FillDiagonal, FillDiagonalCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
