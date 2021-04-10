/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/update_cache_cpu_kernel.h"
#include <string>
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
void UpdateCacheCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  node_wpt_ = kernel_node;

  input_x_dtype_ = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 0);
  indices_dtype_ = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 1);

  if (input_x_dtype_ == kNumberTypeFloat32 || input_x_dtype_ == kNumberTypeInt32) {
    input_x_dtype_size_ = 4;
  } else if (input_x_dtype_ == kNumberTypeFloat64 || input_x_dtype_ == kNumberTypeInt64) {
    input_x_dtype_size_ = 8;
  } else {
    MS_LOG(EXCEPTION) << "input_x dtype only support float32, float64, int32, int64";
  }
}

bool UpdateCacheCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                  const std::vector<kernel::AddressPtr> & /*workspace*/,
                                  const std::vector<kernel::AddressPtr> &outputs) {
  if (indices_dtype_ == kNumberTypeInt32) {
    LaunchKernel<int>(inputs, outputs);
  } else if (indices_dtype_ == kNumberTypeInt64) {
    LaunchKernel<int64_t>(inputs, outputs);
  } else {
    MS_LOG(ERROR) << "indices dtype only support int32, int64";
    return false;
  }
  return true;
}

template <typename T>
void UpdateCacheCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  auto node_ = node_wpt_.lock();
  if (!node_) {
    MS_LOG(EXCEPTION) << "node_wpt_ is expired.";
  }
  auto indices_shape = AnfAlgo::GetPrevNodeOutputInferShape(node_, 1);
  auto update_shape = AnfAlgo::GetPrevNodeOutputInferShape(node_, 2);

  batch_size_ = 1;
  for (size_t i = 0; i < indices_shape.size(); ++i) {
    batch_size_ *= indices_shape[i];
  }
  MS_LOG(INFO) << "UpdateCache batch_size:" << batch_size_;
  update_size_ = 1;
  for (size_t i = 0; i < update_shape.size(); ++i) {
    update_size_ *= update_shape[i];
  }
  update_length_ = update_shape[1];
  char *input_x = reinterpret_cast<char *>(inputs[0]->addr);
  T *indices = reinterpret_cast<T *>(inputs[1]->addr);
  char *update = reinterpret_cast<char *>(inputs[2]->addr);
  max_num_ = *reinterpret_cast<T *>(inputs[3]->addr);

  size_t one_length_size = input_x_dtype_size_ * update_length_;
  auto max_size = inputs[0]->size;
  for (size_t i = 0; i < batch_size_; ++i) {
    if (indices[i] < 0 || indices[i] >= max_num_) continue;

    char *tmp = update + i * one_length_size;
    if (indices[i] * one_length_size + one_length_size <= max_size) {
      int ret =
        memcpy_s(input_x + indices[i] * one_length_size, max_size - indices[i] * one_length_size, tmp, one_length_size);
      if (ret != 0) {
        MS_LOG(EXCEPTION) << "memcpy_s error, errorno" << ret;
      }
    } else {
      MS_LOG(EXCEPTION) << "Memcpy out of size";
    }
  }
}
}  // namespace kernel
}  // namespace mindspore
