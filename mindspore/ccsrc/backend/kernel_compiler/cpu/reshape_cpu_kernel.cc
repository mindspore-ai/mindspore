/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/reshape_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kReshapeInputsNum = 1;
constexpr size_t kReshapeOutputsNum = 1;
}  // namespace

void ReshapeCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  node_wpt_ = kernel_node;
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  x_data_type_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  type_size_ = GetTypeByte(TypeIdToType(x_data_type_));
}

bool ReshapeCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                              const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kReshapeInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kReshapeOutputsNum, kernel_name_);
  if (inputs[0]->size != outputs[0]->size) {
    return false;
  }
  if (inputs[0]->addr == outputs[0]->addr) {
    return true;
  }

  auto node = node_wpt_.lock();
  if (!node) {
    MS_LOG(EXCEPTION) << "node_wpt_ is expired.";
  }
  auto x_shape = AnfAlgo::GetPrevNodeOutputInferShape(node, 0);
  size_t mem_bits = type_size_;
  for (size_t i = 0; i < x_shape.size(); ++i) {
    mem_bits *= x_shape[i];
  }
  auto ret = memcpy_s(outputs[0]->addr, mem_bits, inputs[0]->addr, mem_bits);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno" << ret;
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
