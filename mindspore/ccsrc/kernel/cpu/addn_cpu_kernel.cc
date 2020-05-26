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

#include "kernel/cpu/addn_cpu_kernel.h"
#include "device/cpu/cpu_device_address.h"
#include "ir/primitive.h"

namespace mindspore {
namespace kernel {
void AddNCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  CheckParam(kernel_node);
  input_num_ = AnfAlgo::GetInputTensorNum(kernel_node);
  output_shape_ = AnfAlgo::GetOutputInferShape(kernel_node, 0);
  CPUKernelUtils::ExpandDimsTo4(&output_shape_);
}

bool AddNCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                           const std::vector<kernel::AddressPtr> & /*workspace*/,
                           const std::vector<kernel::AddressPtr> &outputs) {
  auto output_addr = reinterpret_cast<float *>(outputs[0]->addr);

  for (size_t i = 0; i < output_shape_[0]; ++i) {
    for (size_t j = 0; j < output_shape_[1]; ++j) {
      for (size_t k = 0; k < output_shape_[2]; ++k) {
        for (size_t m = 0; m < output_shape_[3]; ++m) {
          auto offset = CPUKernelUtils::CalcOffset(output_shape_, i, j, k, m);
          float sum = 0;
          for (size_t index = 0; index < input_num_; ++index) {
            auto input_addr = reinterpret_cast<float *>(inputs[index]->addr);
            sum += input_addr[offset];
          }
          output_addr[offset] = sum;
        }
      }
    }
  }

  return true;
}

void AddNCPUKernel::CheckParam(const CNodePtr &kernel_node) {
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  if (input_shape.size() > 4) {
    MS_LOG(EXCEPTION) << "Input dims is " << input_shape.size() << ", but AddNCPUKernel olny support 4d or lower.";
  }

  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << "Output number is " << output_num << ", but AddNCPUKernel needs 1 output.";
  }
}
}  // namespace kernel
}  // namespace mindspore
