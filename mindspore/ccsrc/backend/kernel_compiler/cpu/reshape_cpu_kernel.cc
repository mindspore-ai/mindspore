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
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
}

bool ReshapeCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                              const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.empty()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the inputs should be not empty.";
  }
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kReshapeOutputsNum, kernel_name_);
  if (inputs[0]->size != outputs[0]->size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the size of 'input_x': {" << inputs[0]->size
                      << "} is not equal to the size of the first output: {" << outputs[0]->size << "}";
  }
  if (inputs[0]->addr == outputs[0]->addr) {
    return true;
  }
  size_t copy_size = outputs[0]->size;
  auto ret = memcpy_s(outputs[0]->addr, copy_size, inputs[0]->addr, copy_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memcpy_s error. Error no: " << ret;
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
