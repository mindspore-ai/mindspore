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

#include "plugin/device/cpu/kernel/pad_and_shift_cpu_kernel.h"

namespace mindspore {
namespace kernel {
int PadAndShiftCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  input_x_dtype_ = inputs[kIndex0]->dtype_id();
  type_size_ = GetTypeByte(TypeIdToType(input_x_dtype_));
  input_shape_ = inputs[kIndex0]->GetShapeVector();
  batch_size_ = SizeOf(input_shape_);
  MS_LOG(INFO) << "PadAndShift batch_size:" << batch_size_;
  const auto &cum_sum_arr_shape = inputs[kIndex1]->GetShapeVector();
  if (cum_sum_arr_shape.size() != 1) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'cum_sum_arr' must be 1, but got "
                  << cum_sum_arr_shape.size() << ".";
  }
  cum_sum_size_ = LongToSize(cum_sum_arr_shape[0]);
  is_need_retrieve_output_shape_ = true;
  return KRET_OK;
}

bool PadAndShiftCpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                                     const std::vector<kernel::KernelTensor *> &,
                                     const std::vector<kernel::KernelTensor *> &outputs) {
  if (input_x_dtype_ == kNumberTypeInt32) {
    LaunchKernel<int>(inputs, outputs);
  } else if (input_x_dtype_ == kNumberTypeInt64) {
    LaunchKernel<int64_t>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of 'input_x' must be int32 or int64, but got "
                      << input_x_dtype_;
  }
  return true;
}

template <typename T>
void PadAndShiftCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<kernel::KernelTensor *> &outputs) {
  T *input_x = reinterpret_cast<T *>(inputs[0]->device_ptr());
  T *cum_sum_arr = reinterpret_cast<T *>(inputs[1]->device_ptr());
  T shift_idx = *reinterpret_cast<T *>(inputs[2]->device_ptr());
  T *output = reinterpret_cast<T *>(outputs[0]->device_ptr());

  if (shift_idx < 0 || shift_idx >= static_cast<T>(cum_sum_size_)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', shift index must be large than 0 and less than cumsum size, but got shift index: "
                      << shift_idx << " and cumsum size: " << cum_sum_size_;
  }
  output_size_ = static_cast<size_t>(cum_sum_arr[cum_sum_size_ - 1]);
  size_t shift_size = static_cast<size_t>(cum_sum_arr[shift_idx]);
  size_t valid_size = static_cast<size_t>(cum_sum_arr[shift_idx + 1]) - shift_size;
  int ret = memset_s(output, outputs[0]->size(), -1, type_size_ * output_size_);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset_s error. Error no: " << ret;
  }
  ret = memcpy_s(output + shift_size, valid_size * type_size_, input_x, valid_size * type_size_);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memcpy_s error. Error no: " << ret;
  }
}

void PadAndShiftCpuKernelMod::UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                                       const std::vector<KernelTensor *> &outputs) {
  ShapeVector output_shape(input_shape_.begin(), input_shape_.end());
  output_shape[kIndex0] = output_size_;
  outputs[kIndex0]->SetShapeVector(output_shape);
  outputs[kIndex0]->set_size(output_size_ * type_size_);
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, PadAndShift, PadAndShiftCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
