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

#include "backend/kernel_compiler/cpu/map_uniform_cpu_kernel.h"
#include <string>
#include <memory>
#include <vector>
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
void MapUniformCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  node_wpt_ = kernel_node;
  dtype_ = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 0);
}

bool MapUniformCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                 const std::vector<kernel::AddressPtr> & /*workspace*/,
                                 const std::vector<kernel::AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeInt32) {
    LaunchKernel<int>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt64) {
    LaunchKernel<int64_t>(inputs, outputs);
  } else {
    MS_LOG(ERROR) << "Only support int32, int64";
    return false;
  }
  return true;
}

template <typename T>
void MapUniformCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  auto node_ = node_wpt_.lock();
  if (!node_) {
    MS_LOG(EXCEPTION) << "node_wpt_ is expired.";
  }
  auto input_x_shape = AnfAlgo::GetPrevNodeOutputInferShape(node_, 0);
  batch_size_ = 1;
  for (size_t i = 0; i < input_x_shape.size(); ++i) {
    batch_size_ *= input_x_shape[i];
  }
  MS_LOG(INFO) << "Input size: " << batch_size_;
  auto input_x = reinterpret_cast<T *>(inputs[0]->addr);
  auto per_group_size = *reinterpret_cast<T *>(inputs[1]->addr);
  auto group_num = *reinterpret_cast<T *>(inputs[2]->addr);
  auto output_x = reinterpret_cast<T *>(outputs[0]->addr);
  T max_num = group_num * per_group_size;
  for (size_t i = 0; i < batch_size_; ++i) {
    output_x[i] = input_x[i] % group_num * per_group_size + input_x[i] / group_num;
    if (output_x[i] >= max_num) {
      MS_LOG(EXCEPTION) << "Value can not >= " << max_num;
    }
  }
}
}  // namespace kernel
}  // namespace mindspore
