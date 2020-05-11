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
#include "kernel/cpu/slice_grad_cpu_kernel.h"
#include "device/cpu/cpu_device_address.h"
#include "ir/primitive.h"

namespace mindspore {
namespace kernel {
void SliceGradCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  CheckParam(kernel_node);

  begin_ = AnfAlgo::GetNodeAttr<std::vector<int>>(kernel_node, BEGIN);
  size_ = AnfAlgo::GetNodeAttr<std::vector<int>>(kernel_node, SIZE);

  input_dy_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  if (input_dy_shape_.size() < 4) {
    for (size_t i = 0; i < 4 - input_dy_shape_.size(); ++i) {
      input_dy_shape_.insert(input_dy_shape_.begin(), 1);
      begin_.insert(begin_.begin(), 0);
      size_.insert(size_.begin(), 1);
    }
  }

  input_x_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  output_dx_shape_ = AnfAlgo::GetOutputInferShape(kernel_node, 0);
  CPUKernelUtils::ExpandDimsTo4(&input_x_shape_);
  CPUKernelUtils::ExpandDimsTo4(&output_dx_shape_);

  for (size_t i = 0; i < begin_.size(); i++) {
    if (begin_[i] < 0) {
      begin_[i] = begin_[i] + input_x_shape_[i];
    }
  }

  for (size_t i = 0; i < size_.size(); i++) {
    if (size_[i] < 0) {
      size_[i] = (size_[i] + input_x_shape_[i]) > 0 ? (size_[i] + input_x_shape_[i]) : 0;
    }
  }
}

bool SliceGradCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                const std::vector<kernel::AddressPtr> & /*workspace*/,
                                const std::vector<kernel::AddressPtr> &outputs) {
  auto input_dy_addr = reinterpret_cast<float *>(inputs[0]->addr);
  auto output_dx_addr = reinterpret_cast<float *>(outputs[0]->addr);

  auto out_size = sizeof(float) * output_dx_shape_[0] * output_dx_shape_[1] * output_dx_shape_[2] * output_dx_shape_[3];
  auto ret = memset_s(output_dx_addr, out_size, 0, out_size);
  if (ret != EOK) {
    MS_LOG(ERROR) << "output buff memset fail.";
    return false;
  }

  for (int i = begin_[0]; i < begin_[0] + size_[0]; ++i) {
    for (int j = begin_[1]; j < begin_[1] + size_[1]; ++j) {
      for (int k = begin_[2]; k < begin_[2] + size_[2]; ++k) {
        for (int m = begin_[3]; m < begin_[3] + size_[3]; ++m) {
          auto offset = CPUKernelUtils::CalcOffset(output_dx_shape_, i, j, k, m);
          output_dx_addr[offset] = *input_dy_addr++;
        }
      }
    }
  }
  return true;
}

void SliceGradCPUKernel::CheckParam(const CNodePtr &kernel_node) {
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << "Output number is " << output_num << ", but SliceGradGpuKernel needs 1 output.";
  }
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  if (input_shape.size() > 4) {
    MS_LOG(EXCEPTION) << "Input dims is " << input_shape.size() << ", but SliceGradGpuKernel only support 4d or lower.";
  }
  if (input_shape.size() == 0) {
    MS_LOG(EXCEPTION) << "Input dims is " << input_shape.size() << ", scalar is not supported.";
  }
}
}  // namespace kernel
}  // namespace mindspore
