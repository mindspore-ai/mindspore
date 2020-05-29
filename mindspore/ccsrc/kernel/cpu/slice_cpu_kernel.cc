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
#include "kernel/cpu/slice_cpu_kernel.h"
#include "device/cpu/cpu_device_address.h"
#include "ir/primitive.h"

namespace mindspore {
namespace kernel {
void SliceCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  CheckParam(kernel_node);
  input_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  output_shape_ = AnfAlgo::GetOutputInferShape(kernel_node, 0);
  CPUKernelUtils::ExpandDimsTo4(&output_shape_);

  begin_ = AnfAlgo::GetNodeAttr<std::vector<int>>(kernel_node, BEGIN);
  for (size_t i = 0; i < begin_.size(); i++) {
    if (begin_[i] < 0) {
      begin_[i] = begin_[i] + input_shape_[i];
    }
  }
  auto prim = AnfAlgo::GetCNodePrimitive(kernel_node);
  MS_EXCEPTION_IF_NULL(prim);
  auto strides = prim->GetAttr(STRIDES);
  if (strides != nullptr) {
    strides_ = AnfAlgo::GetNodeAttr<std::vector<int>>(kernel_node, STRIDES);
    end_ = AnfAlgo::GetNodeAttr<std::vector<int>>(kernel_node, END);
    if (strides_.size() != end_.size() || strides_.size() != input_shape_.size()) {
      MS_LOG(EXCEPTION) << "stride|end|input size must be equal";
    }
    for (size_t i = 0; i < strides_.size(); ++i) {
      if (strides_[i] < 0) {
        strides_[i] = (strides_[i] + input_shape_[i]) > 0 ? (strides_[i] + input_shape_[i]) : 0;
      }
      if (end_[i] < 0) {
        end_[i] = (end_[i] + input_shape_[i]) > 0 ? (end_[i] + input_shape_[i]) : 0;
      }
    }
  } else {
    auto sizes = AnfAlgo::GetNodeAttr<std::vector<int>>(kernel_node, SIZE);
    if (sizes.size() != input_shape_.size() || begin_.size() != input_shape_.size()) {
      MS_LOG(EXCEPTION) << "begin|size|input size must be equal";
    }
    for (size_t i = 0; i < sizes.size(); ++i) {
      if (sizes[i] < 0) {
        sizes[i] = (sizes[i] + input_shape_[i]) > 0 ? (sizes[i] + input_shape_[i]) : 0;
      }
      strides_.emplace_back(1);
      end_.emplace_back(begin_[i] + sizes[i]);
    }
  }
  auto input_len = input_shape_.size();
  if (input_len < 4) {
    for (size_t i = 0; i < 4 - input_len; ++i) {
      input_shape_.insert(input_shape_.begin(), 1);
      begin_.insert(begin_.begin(), 0);
      strides_.insert(strides_.begin(), 1);
      end_.insert(end_.begin(), 1);
    }
  }
}

bool SliceCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                            const std::vector<kernel::AddressPtr> & /*workspace*/,
                            const std::vector<kernel::AddressPtr> &outputs) {
  auto input_addr = reinterpret_cast<float *>(inputs[0]->addr);
  auto output_addr = reinterpret_cast<float *>(outputs[0]->addr);

  for (int i = begin_[0]; i < end_[0]; i += strides_[0]) {
    for (int j = begin_[1]; j < end_[1]; j += strides_[1]) {
      for (int k = begin_[2]; k < end_[2]; k += strides_[2]) {
        for (int m = begin_[3]; m < end_[3]; m += strides_[3]) {
          auto offset = CPUKernelUtils::CalcOffset(input_shape_, i, j, k, m);
          *output_addr++ = input_addr[offset];
        }
      }
    }
  }

  return true;
}

void SliceCPUKernel::CheckParam(const CNodePtr &kernel_node) {
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 1) {
    MS_LOG(EXCEPTION) << "Input number is " << input_num << ", but SliceCPUKernel needs 1 inputs.";
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << "Output number is " << output_num << ", but SliceCPUKernel needs 1 output.";
  }
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  if (input_shape.size() > 4) {
    MS_LOG(EXCEPTION) << "Input dims is " << input_shape.size() << ", but SliceCPUKernel olny support 4d or lower.";
  }
  if (input_shape.size() == 0) {
    MS_LOG(EXCEPTION) << "Input dims is " << input_shape.size() << ", scalar is not supported.";
  }
}
}  // namespace kernel
}  // namespace mindspore
