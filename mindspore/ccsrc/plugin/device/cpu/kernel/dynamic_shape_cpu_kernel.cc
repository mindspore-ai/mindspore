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

#include "plugin/device/cpu/kernel/dynamic_shape_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kDynamicShapeOutputNum = 1;
}  // namespace
void TensorShapeCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  cnode_ptr_ = kernel_node;
  size_t input_count = common::AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_count != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be 1, but got " << input_count;
  }
}

bool TensorShapeCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                     const std::vector<kernel::AddressPtr> &,
                                     const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kDynamicShapeOutputNum, kernel_name_);
  auto node_ = cnode_ptr_.lock();
  if (node_ == nullptr) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', cnode_ptr_(kernel_node) is expired. Error no: " << node_;
  }
  auto output_addr = reinterpret_cast<int64_t *>(outputs[0]->addr);
  std::vector<size_t> input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(node_, 0);
  auto output_shape = common::AnfAlgo::GetOutputInferShape(node_, 0);
  if (output_shape.size() != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of output should be 1-D, but got: " << output_shape.size();
  }
  if (output_shape[0] != input_shape.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', 'output_shape[0]' should be equal to the dimension of input, but got 'output_shape[0]': "
                      << output_shape[0] << " and the dimension of input: " << input_shape.size();
  }
  for (size_t i = 0; i < output_shape[0]; ++i) {
    output_addr[i] = input_shape[i];
  }
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, DynamicShape, TensorShapeCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, TensorShape, TensorShapeCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
