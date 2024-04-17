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

#include "plugin/device/cpu/kernel/update_cache_cpu_kernel.h"
#include <string>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMinUpdateShapeSize = 2;
}  // namespace

int UpdateCacheCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  input_x_dtype_ = inputs[kIndex0]->dtype_id();
  indices_dtype_ = inputs[kIndex1]->dtype_id();

  if (input_x_dtype_ == kNumberTypeFloat32 || input_x_dtype_ == kNumberTypeInt32) {
    input_x_dtype_size_ = 4;
  } else if (input_x_dtype_ == kNumberTypeFloat64 || input_x_dtype_ == kNumberTypeInt64) {
    input_x_dtype_size_ = 8;
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dtype of 'input_x' must be float32, float64, int32, int64, but got: " << input_x_dtype_;
  }
  return KRET_OK;
}

bool UpdateCacheCpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                                     const std::vector<kernel::KernelTensor *> &,
                                     const std::vector<kernel::KernelTensor *> &outputs) {
  if (indices_dtype_ == kNumberTypeInt32) {
    LaunchKernel<int>(inputs, outputs);
  } else if (indices_dtype_ == kNumberTypeInt64) {
    LaunchKernel<int64_t>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dtype of 'indices' must be int32 or int64, but got: " << indices_dtype_;
  }
  return true;
}

template <typename T>
void UpdateCacheCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<kernel::KernelTensor *> &) {
  const auto &indices_shape = inputs[kIndex1]->GetShapeVector();
  const auto &update_shape = inputs[kIndex2]->GetShapeVector();
  if (AnfAlgo::IsShapesDynamic({indices_shape, update_shape})) {
    return;
  }
  if (update_shape.size() < kMinUpdateShapeSize) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of 'update' must be at least "
                      << kMinUpdateShapeSize << "D, but got: " << update_shape.size() << "D";
  }
  batch_size_ = SizeOf(indices_shape);
  MS_LOG(INFO) << "UpdateCache batch_size:" << batch_size_;
  update_size_ = SizeToLong(SizeOf(update_shape));
  update_length_ = LongToSize(update_shape[1]);
  char *input_x = reinterpret_cast<char *>(inputs[kIndex0]->device_ptr());
  T *indices = reinterpret_cast<T *>(inputs[kIndex1]->device_ptr());
  char *update = reinterpret_cast<char *>(inputs[kIndex2]->device_ptr());
  auto max_num = *reinterpret_cast<T *>(inputs[kIndex3]->device_ptr());

  size_t one_length_size = input_x_dtype_size_ * update_length_;
  auto max_size = inputs[kIndex0]->size();
  for (size_t i = 0; i < batch_size_; ++i) {
    if (indices[i] < 0 || indices[i] >= max_num) {
      continue;
    }
    char *tmp = update + i * one_length_size;
    if (static_cast<size_t>(indices[i]) * one_length_size + one_length_size > max_size) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memcpy out of size.";
    }
    int ret = memcpy_s(input_x + static_cast<size_t>(indices[i]) * one_length_size,
                       max_size - static_cast<size_t>(indices[i]) * one_length_size, tmp, one_length_size);
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memcpy_s error. Error no: " << ret;
    }
  }
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, UpdateCache, UpdateCacheCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
