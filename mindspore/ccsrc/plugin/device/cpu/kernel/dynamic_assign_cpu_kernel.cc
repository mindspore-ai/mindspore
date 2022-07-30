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
namespace {
constexpr size_t kDynamicAssignInputsNum = 2;
constexpr size_t kDynamicAssignOutputsNum = 1;
}  // namespace

void DynamicAssignCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  node_wpt_ = kernel_node;
  input_x_dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  input_x_dtype_size_ = GetTypeByte(TypeIdToType(input_x_dtype_));
}

bool DynamicAssignCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
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
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << ", the dtype of 'input_x' must be in (int32, int64, float32, float64) on CPU, but got "
                      << TypeIdToType(input_x_dtype_)->ToString();
  }
  return true;
}

template <typename T>
void DynamicAssignCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                             const std::vector<kernel::AddressPtr> &) {
  auto node = node_wpt_.lock();
  if (!node) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', node_wpt_(kernel_node) is expired. Error no: " << node;
  }
  auto input_x_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(node, 0);
  auto input_y_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(node, 1);
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
                        << Vector2Str(input_x_shape) << ", and the shape of 'input_y': " << Vector2Str(input_y_shape);
    }
  }
  auto *input_x = reinterpret_cast<T *>(inputs[0]->addr);
  auto *input_y = reinterpret_cast<T *>(inputs[1]->addr);
  auto max_size = inputs[0]->size;
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

  auto node_with_idx = common::AnfAlgo::GetPrevNodeOutput(node, 0);
  auto out_node = node_with_idx.first;
  if (out_node->isa<Parameter>()) {
    auto node_ptr = out_node->cast<ParameterPtr>();
    auto value = node_ptr->default_param();
    auto tensor = value->cast<std::shared_ptr<tensor::Tensor>>();
    (void)tensor->set_shape(input_x_shape);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', output must be a Parameter.";
  }
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
