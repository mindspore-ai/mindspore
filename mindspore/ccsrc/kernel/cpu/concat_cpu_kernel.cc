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

#include "kernel/cpu/concat_cpu_kernel.h"
#include "device/cpu/cpu_device_address.h"
#include "ir/primitive.h"

namespace mindspore {
namespace kernel {
void ConcatCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  CheckParam(kernel_node);

  axis_ = AnfAlgo::GetNodeAttr<int>(kernel_node, AXIS);
  auto input_1_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  if (axis_ < 0) {
    axis_ = axis_ + SizeToInt(input_1_shape.size());
  }
  axis_ += 4 - input_1_shape.size();

  auto input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  for (size_t i = 0; i < input_num; i++) {
    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, i);
    CPUKernelUtils::ExpandDimsTo4(&input_shape);
    input_shape_list_.push_back(input_shape);
  }

  output_shape_ = AnfAlgo::GetOutputInferShape(kernel_node, 0);
  CPUKernelUtils::ExpandDimsTo4(&output_shape_);
}

bool ConcatCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                             const std::vector<kernel::AddressPtr> & /*workspace*/,
                             const std::vector<kernel::AddressPtr> &outputs) {
  auto output_addr = reinterpret_cast<float *>(outputs[0]->addr);
  auto buff_size = outputs[0]->size;
  size_t dim0 = output_shape_[0];
  size_t dim1 = output_shape_[1];
  size_t dim2 = output_shape_[2];

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

void ConcatCPUKernel::CopyDataToOutput(const std::vector<kernel::AddressPtr> &inputs, size_t dim0, size_t dim1,
                                       size_t dim2, float **output_addr, size_t *buff_size) {
  for (size_t i = 0; i < input_shape_list_.size(); ++i) {
    auto input_i_shape = input_shape_list_[i];
    auto input_i_addr = reinterpret_cast<float *>(inputs[i]->addr);

    size_t num = CPUKernelUtils::GetElementNumOnAxis(input_i_shape, axis_);
    num *= input_i_shape[axis_];
    auto pos = CPUKernelUtils::CalcOffset(input_i_shape, dim0, dim1, dim2, 0);
    auto ret = memcpy_s(*output_addr, *buff_size, input_i_addr + pos, num * sizeof(float));
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "memcpy failed.";
    }
    *output_addr += num;
    *buff_size -= num * sizeof(float);
  }
}

void ConcatCPUKernel::CheckParam(const CNodePtr &kernel_node) {
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  if (input_shape.size() > 4) {
    MS_LOG(EXCEPTION) << "Input dims is " << input_shape.size() << ", but ConcatCPUKernel olny support 4d or lower.";
  }

  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << "Output number is " << output_num << ", but ConcatCPUKernel needs 1 output.";
  }
}
}  // namespace kernel
}  // namespace mindspore
