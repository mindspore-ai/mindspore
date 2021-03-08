/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "backend/kernel_compiler/cpu/argmax_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
size_t get_element_num(const std::vector<size_t> &shape) {
  size_t size = 1;
  for (size_t i = 0; i < shape.size(); i++) {
    size *= shape[i];
  }
  return size;
}

template <typename T>
bool check_validation(const std::vector<size_t> &shape, const size_t num_before_axis, const size_t num_after_axis,
                      const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.size() != 1 || outputs.size() != 1) {
    MS_LOG(EXCEPTION) << "Wrong number of inputs or outputs!";
    return false;
  }
  size_t data_size = sizeof(T);
  size_t input_size = get_element_num(shape) * data_size;
  size_t output_num = num_before_axis * num_after_axis;
  size_t output_size = output_num * sizeof(int);
  if (inputs[0]->size != input_size || outputs[0]->size != output_size) {
    MS_LOG(EXCEPTION) << "invalid input or output data size!";
    return false;
  }
  return true;
}
}  // namespace

template <typename T>
void ArgmaxCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  size_t shape_len = shape_.size();
  int64_t axis = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, AXIS);
  axis += shape_len;
  if (axis < 0) {
    MS_LOG(EXCEPTION) << "Invalid axis:" << axis << ", should in range [-1, " << shape_len - 1 << "]";
  }
  axis = axis % static_cast<int64_t>(shape_len);
  num_before_axis_ = 1;
  num_after_axis_ = 1;
  for (size_t i = 0; i < shape_len; i++) {
    if (static_cast<int64_t>(i) < axis) {
      num_before_axis_ *= shape_[i];
    } else if (static_cast<int64_t>(i) > axis) {
      num_after_axis_ *= shape_[i];
    }
  }
  dim_axis_ = shape_[axis];
}

template <typename T>
bool ArgmaxCPUKernel<T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                const std::vector<kernel::AddressPtr> & /*workspaces*/,
                                const std::vector<kernel::AddressPtr> &outputs) {
  if (!check_validation<T>(shape_, num_before_axis_, num_after_axis_, inputs, outputs)) {
    return false;
  }

  auto input = reinterpret_cast<T *>(inputs[0]->addr);
  auto output = reinterpret_cast<int32_t *>(outputs[0]->addr);

  for (size_t i = 0; i < num_before_axis_; i++) {
    size_t src_index_i = i * dim_axis_ * num_after_axis_;
    for (size_t j = 0; j < num_after_axis_; j++) {
      std::vector<float> array_axis;
      size_t src_index_j = src_index_i + j;
      for (size_t k = 0; k < dim_axis_; k++) {
        size_t src_index_k = k * num_after_axis_ + src_index_j;
        array_axis.push_back(static_cast<float>(input[src_index_k]));
      }
      auto max_ops = std::max_element(array_axis.begin(), array_axis.end());
      auto max_index = static_cast<int32_t>(std::distance(array_axis.begin(), max_ops));
      auto dst_index = i * num_after_axis_ + j;
      output[dst_index] = max_index;
    }
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
