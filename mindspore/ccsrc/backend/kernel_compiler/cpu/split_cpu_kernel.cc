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

#include "backend/kernel_compiler/cpu/split_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
void SplitCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  CheckParam(kernel_node);

  axis_ = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, AXIS);
  auto output_1_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
  if (axis_ < 0) {
    axis_ = axis_ + SizeToLong(output_1_shape.size());
  }
  axis_ += 4 - SizeToLong(output_1_shape.size());

  auto output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  for (size_t i = 0; i < output_num; i++) {
    auto output_shape = AnfAlgo::GetOutputInferShape(kernel_node, i);
    CPUKernelUtils::ExpandDimsTo4(&output_shape);
    output_shape_list_.push_back(output_shape);
  }

  input_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  CPUKernelUtils::ExpandDimsTo4(&input_shape_);

  dtype_ = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 0);
}

bool SplitCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                            const std::vector<kernel::AddressPtr> & /*workspace*/,
                            const std::vector<kernel::AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeInt32 || dtype_ == kNumberTypeInt) {
    return LaunchKernel<int32_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt64) {
    return LaunchKernel<int64_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32 || dtype_ == kNumberTypeFloat) {
    return LaunchKernel<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat64) {
    return LaunchKernel<double>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "Only support int, float, but actual data type is " << TypeIdLabel(dtype_);
  }
}

template <typename T>
bool SplitCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  auto input_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto buff_size = inputs[0]->size;
  size_t dim0 = input_shape_[0];
  size_t dim1 = input_shape_[1];
  size_t dim2 = input_shape_[2];

  if (axis_ == 3) {
    for (size_t i = 0; i < dim0; ++i) {
      for (size_t j = 0; j < dim1; ++j) {
        for (size_t k = 0; k < dim2; ++k) {
          CopyDataToOutput(outputs, i, j, k, &input_addr, &buff_size);
        }
      }
    }
  } else if (axis_ == 2) {
    for (size_t i = 0; i < dim0; ++i) {
      for (size_t j = 0; j < dim1; ++j) {
        CopyDataToOutput(outputs, i, j, 0, &input_addr, &buff_size);
      }
    }
  } else if (axis_ == 1) {
    for (size_t i = 0; i < dim0; ++i) {
      CopyDataToOutput(outputs, i, 0, 0, &input_addr, &buff_size);
    }
  } else if (axis_ == 0) {
    CopyDataToOutput(outputs, 0, 0, 0, &input_addr, &buff_size);
  }
  return true;
}

template <typename T>
void SplitCPUKernel::CopyDataToOutput(const std::vector<kernel::AddressPtr> &outputs, size_t dim0, size_t dim1,
                                      size_t dim2, T **input_addr, size_t *buff_size) {
  for (size_t i = 0; i < output_shape_list_.size(); ++i) {
    auto output_i_shape = output_shape_list_[i];
    auto output_i_addr = reinterpret_cast<float *>(outputs[i]->addr);

    size_t num = CPUKernelUtils::GetElementNumOnAxis(output_i_shape, axis_);
    num *= output_i_shape[axis_];
    auto pos = CPUKernelUtils::CalcOffset(output_i_shape, dim0, dim1, dim2, 0);
    auto ret = memcpy_s(output_i_addr + pos, *buff_size, *input_addr, num * sizeof(T));
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "memcpy failed.";
    }
    *input_addr += num;
    *buff_size -= num * sizeof(T);
  }
}

void SplitCPUKernel::CheckParam(const CNodePtr &kernel_node) {
  auto output_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
  if (output_shape.size() > 4) {
    MS_LOG(EXCEPTION) << "Output dims is " << output_shape.size() << ", but SplitCPUKernel only support 4d or lower.";
  }

  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 1) {
    MS_LOG(EXCEPTION) << "Input number is " << input_num << ", but SplitCPUKernel needs 1 input.";
  }
}
}  // namespace kernel
}  // namespace mindspore
