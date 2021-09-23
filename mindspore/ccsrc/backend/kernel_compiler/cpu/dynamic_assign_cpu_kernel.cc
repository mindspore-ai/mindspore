/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/dynamic_assign_cpu_kernel.h"

#include <algorithm>

#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kDynamicAssignInputsNum = 2;
constexpr size_t kDynamicAssignOutputsNum = 1;
}  // namespace

void DynamicAssignCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  node_wpt_ = kernel_node;
  input_x_dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  input_x_dtype_size_ = GetTypeByte(TypeIdToType(input_x_dtype_));
}

bool DynamicAssignCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                    const std::vector<kernel::AddressPtr> &,
                                    const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kDynamicAssignInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kDynamicAssignOutputsNum, kernel_name_);
  if (input_x_dtype_ == kNumberTypeInt32) {
    LaunchKernel<int>(inputs, outputs);
  } else if (input_x_dtype_ == kNumberTypeInt64) {
    LaunchKernel<int64_t>(inputs, outputs);
  } else if (input_x_dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  } else if (input_x_dtype_ == kNumberTypeFloat64) {
    LaunchKernel<double>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << kernel_name_ << " support (int32, int64, float32, float64) on CPU , but got "
                      << TypeIdToType(input_x_dtype_)->ToString();
  }
  return true;
}

template <typename T>
void DynamicAssignCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                          const std::vector<kernel::AddressPtr> &outputs) {
  auto node = node_wpt_.lock();
  if (!node) {
    MS_LOG(EXCEPTION) << kernel_name_ << " node_wpt_ is expired.";
  }
  auto input_x_shape = AnfAlgo::GetPrevNodeOutputInferShape(node, 0);
  auto input_y_shape = AnfAlgo::GetPrevNodeOutputInferShape(node, 1);
  batch_size_ = 1;
  for (size_t i = 0; i < input_x_shape.size(); ++i) {
    batch_size_ *= input_x_shape[i];
  }

  if (input_x_shape.size() != input_y_shape.size()) {
    MS_LOG(EXCEPTION) << "X and y must be same shape";
  }
  for (size_t i = 0; i < input_x_shape.size(); ++i) {
    if (input_x_shape[i] != input_y_shape[i]) {
      MS_LOG(EXCEPTION) << "x and y must be same shape!";
    }
  }
  auto *input_x = reinterpret_cast<T *>(inputs[0]->addr);
  auto *input_y = reinterpret_cast<T *>(inputs[1]->addr);
  auto max_size = inputs[0]->size;
  size_t total_size = input_x_dtype_size_ * batch_size_;
  if (total_size > max_size) {
    MS_LOG(EXCEPTION) << "Memcpy size must <= max_size, but got memcpy size is : " << total_size
                      << ", max size is : " << max_size;
  }
  int ret = memcpy_s(input_x, total_size, input_y, total_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "Memcpy_s error, errorno" << ret;
  }

  auto node_with_idx = AnfAlgo::GetPrevNodeOutput(node, 0);
  auto out_node = node_with_idx.first;
  if (out_node->isa<Parameter>()) {
    auto node_ptr = out_node->cast<ParameterPtr>();
    auto value = node_ptr->default_param();
    auto tensor = value->cast<std::shared_ptr<tensor::Tensor>>();
    ShapeVector shape_tmp;
    (void)std::transform(input_x_shape.begin(), input_x_shape.end(), std::back_inserter(shape_tmp), SizeToLong);
    tensor->set_shape(shape_tmp);
  } else {
    MS_LOG(EXCEPTION) << "Input x must be a Parameter.";
  }
}
}  // namespace kernel
}  // namespace mindspore
