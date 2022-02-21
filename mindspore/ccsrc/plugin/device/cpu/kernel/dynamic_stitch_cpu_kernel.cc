/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/dynamic_stitch_cpu_kernel.h"
#include <functional>
#include <algorithm>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kDynamicStitchOutputNum = 1;
}  // namespace

template <typename T>
void DynamicStitchCpuKernelMod<T>::InitKernel(const CNodePtr &kernel_node) {
  cnode_ptr_ = kernel_node;
}

size_t GetShapeSize(const std::vector<size_t> &shape) {
  return std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
}

template <typename T>
void DynamicStitchCpuKernelMod<T>::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kDynamicStitchOutputNum, kernel_name_);
  auto node_ = cnode_ptr_.lock();
  int first_dim_size = 0;
  size_t input_count = common::AnfAlgo::GetInputTensorNum(node_);
  input_tuple_num_ = input_count / 2;
  int max_index = -1;
  for (size_t i = 0; i < input_tuple_num_; ++i) {
    auto indice = reinterpret_cast<int32_t *>(inputs[i]->addr);
    auto shape_size = GetShapeSize(common::AnfAlgo::GetPrevNodeOutputInferShape(node_, i));
    for (size_t j = 0; j < shape_size; ++j) {
      max_index = std::max(indice[j], max_index);
    }
  }
  first_dim_size = max_index + 1;

  std::vector<TypeId> dtypes{AnfAlgo::GetOutputDeviceDataType(node_, 0)};
  std::vector<size_t> result_shape{IntToSize(first_dim_size)};
  auto data0_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(node_, input_tuple_num_);
  auto indice_dims = common::AnfAlgo::GetPrevNodeOutputInferShape(node_, 0).size();
  for (size_t d = indice_dims; d < data0_shape.size(); ++d) {
    result_shape.emplace_back(data0_shape[d]);
  }
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, {result_shape}, node_.get());

  size_t num_out_dims = 2;
  std::vector<size_t> out_dims(num_out_dims, 0);
  for (size_t out_dim = 0; out_dim <= num_out_dims - 1; ++out_dim) {
    out_dims[out_dim] = out_dim >= result_shape.size() ? 1 : result_shape[out_dim];
  }
  for (size_t in_dim = num_out_dims; in_dim < result_shape.size(); ++in_dim) {
    out_dims[num_out_dims - 1] *= result_shape[in_dim];
  }

  auto merged = reinterpret_cast<T *>(outputs[0]->addr);
  size_t slice_size = out_dims[1];
  size_t slice_bytes = slice_size * sizeof(T);
  for (size_t i = 0; i < input_tuple_num_; i++) {
    auto indice = reinterpret_cast<int32_t *>(inputs[i]->addr);
    auto data = reinterpret_cast<T *>(inputs[i + input_tuple_num_]->addr);
    auto shape_size = GetShapeSize(common::AnfAlgo::GetPrevNodeOutputInferShape(node_, i));
    for (size_t j = 0; j < shape_size; ++j) {
      auto ret = memcpy_s(merged + indice[j] * slice_size, slice_bytes, data + j * slice_size, slice_bytes);
      if (ret != 0) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memcpy_s error. Error no: " << ret;
      }
    }
  }
}

template <typename T>
bool DynamicStitchCpuKernelMod<T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                          const std::vector<kernel::AddressPtr> &,
                                          const std::vector<kernel::AddressPtr> &outputs) {
  LaunchKernel(inputs, outputs);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
