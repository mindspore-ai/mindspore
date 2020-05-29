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
#include "kernel/cpu/gather_cpu_kernel.h"
#include "device/cpu/cpu_device_address.h"
#include "ir/primitive.h"

namespace mindspore {
namespace kernel {
void GatherV2CPUKernel::InitKernel(const CNodePtr &kernel_node) {
  CheckParam(kernel_node);

  input_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  indices_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  output_shape_ = AnfAlgo::GetOutputInferShape(kernel_node, 0);

  axis_ = AnfAlgo::GetNodeAttr<int>(kernel_node, AXIS);
  if (axis_ < 0) {
    axis_ = axis_ + SizeToInt(input_shape_.size());
  }
  axis_ += 4 - input_shape_.size();

  CPUKernelUtils::ExpandDimsTo4(&input_shape_);
  CPUKernelUtils::ExpandDimsTo4(&output_shape_);
}

bool GatherV2CPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                               const std::vector<kernel::AddressPtr> & /*workspace*/,
                               const std::vector<kernel::AddressPtr> &outputs) {
  auto output_addr = reinterpret_cast<float *>(outputs[0]->addr);
  auto buff_size = outputs[0]->size;
  size_t dim0 = input_shape_[0];
  size_t dim1 = input_shape_[1];
  size_t dim2 = input_shape_[2];

  if (axis_ == 3) {
    for (size_t i = 0; i < dim0; ++i) {
      for (size_t j = 0; j < dim1; ++j) {
        for (size_t k = 0; k < dim2; ++k) {
          CopyDataToOutput(inputs, i, j, k, &output_addr, &buff_size);
        }
      }
    }
  } else if (axis_ == 2) {
    for (size_t i = 0; i < dim0; ++i) {
      for (size_t j = 0; j < dim1; ++j) {
        CopyDataToOutput(inputs, i, j, 0, &output_addr, &buff_size);
      }
    }
  } else if (axis_ == 1) {
    for (size_t i = 0; i < dim0; ++i) {
      CopyDataToOutput(inputs, i, 0, 0, &output_addr, &buff_size);
    }
  } else if (axis_ == 0) {
    CopyDataToOutput(inputs, 0, 0, 0, &output_addr, &buff_size);
  }

  return true;
}

void GatherV2CPUKernel::CopyDataToOutput(const std::vector<kernel::AddressPtr> &inputs, size_t dim0, size_t dim1,
                                         size_t dim2, float **output_addr, size_t *buff_size) {
  auto input_addr = reinterpret_cast<float *>(inputs[0]->addr);
  auto indices_addr = reinterpret_cast<int *>(inputs[1]->addr);

  for (size_t i = 0; i < output_shape_[axis_]; ++i) {
    size_t index = IntToSize(indices_addr[i]);
    size_t pos = 0;
    if (axis_ == 3) {
      pos = CPUKernelUtils::CalcOffset(input_shape_, dim0, dim1, dim2, index);
    } else if (axis_ == 2) {
      pos = CPUKernelUtils::CalcOffset(input_shape_, dim0, dim1, index, 0);
    } else if (axis_ == 1) {
      pos = CPUKernelUtils::CalcOffset(input_shape_, dim0, index, 0, 0);
    } else if (axis_ == 0) {
      pos = CPUKernelUtils::CalcOffset(input_shape_, index, 0, 0, 0);
    }
    size_t num = CPUKernelUtils::GetElementNumOnAxis(input_shape_, axis_);
    auto ret = memcpy_s(*output_addr, *buff_size, input_addr + pos, num * sizeof(float));
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "memcpy failed.";
    }
    *output_addr += num;
    *buff_size -= num * sizeof(float);
  }
}

void GatherV2CPUKernel::CheckParam(const CNodePtr &kernel_node) {
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  if (input_shape.size() > 4) {
    MS_LOG(EXCEPTION) << "Input dims is " << input_shape.size() << ", but GatherV2CPUKernel olny support 4d or lower.";
  }

  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 2) {
    MS_LOG(EXCEPTION) << "Argument number is " << input_num << ", but GatherV2CPUKernel needs 2.";
  }
}
}  // namespace kernel
}  // namespace mindspore
