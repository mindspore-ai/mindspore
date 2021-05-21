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
#include "backend/kernel_compiler/cpu/gather_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"
#include "nnacl/gather_parameter.h"
#include "nnacl/base/gather_base.h"

namespace mindspore {
namespace kernel {
template <typename T>
void GatherV2CPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  CheckParam(kernel_node);
  input_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  indices_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  output_shape_ = AnfAlgo::GetOutputInferShape(kernel_node, 0);
  axis_ = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, AXIS);
  if (axis_ < 0) {
    axis_ = axis_ + SizeToLong(input_shape_.size());
  }
  axis_ += 4 - input_shape_.size();
  CPUKernelUtils::ExpandDimsTo4(&input_shape_);
  CPUKernelUtils::ExpandDimsTo4(&output_shape_);
}

template <typename T>
int GatherV2CPUKernel<T>::GatherLaunch(int8_t *input_data, int8_t *output_data, size_t size) {
  int in_rank = input_shape_.size();
  int indices_element_size = 1;
  const int limit = input_shape_.at(axis_);
  size_t data_size = sizeof(kNumberTypeFloat32);
  int outer_size = 1, inner_size = 1;

  for (int i = 0; i < axis_; ++i) {
    outer_size *= input_shape_.at(i);
  }
  for (int i = axis_ + 1; i < in_rank; ++i) {
    inner_size *= input_shape_.at(i);
  }
  for (size_t i = 0; i < indices_shape_.size(); i++) {
    indices_element_size *= indices_shape_.at(i);
  }
  int stride = UP_DIV(outer_size, size);

  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      int8_t *int8_in = input_data;
      int8_t *int8_out = output_data;
      int count = MSMIN(stride, static_cast<int>(outer_size - stride * i));
      if (count <= 0) {
        return;
      }
      auto thread_stride = stride * i;
      int8_in += thread_stride * limit * inner_size * data_size;
      int8_out += thread_stride * indices_element_size * inner_size * data_size;
      auto error_code =
        Gather(int8_in, count, inner_size, limit, indices_data_, indices_element_size, int8_out, sizeof(T));
      if (error_code != 0) {
        MS_LOG(ERROR) << "GatherRun error task_id[" << i << "] error_code[" << error_code << "]";
      }
    }
  };
  CPUKernelUtils::ParallelFor(task, size);

  return 0;
}

template <typename T>
bool GatherV2CPUKernel<T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                  const std::vector<kernel::AddressPtr> & /*workspace*/,
                                  const std::vector<kernel::AddressPtr> &outputs) {
  int8_t *input_tensor = reinterpret_cast<int8_t *>(inputs[0]->addr);
  indices_data_ = reinterpret_cast<int32_t *>(inputs[1]->addr);
  int8_t *output_addr = reinterpret_cast<int8_t *>(outputs[0]->addr);
  size_t size = (outputs[0]->size > 0) ? static_cast<size_t>(outputs[0]->size / sizeof(int8_t)) : 1;

  GatherLaunch(input_tensor, output_addr, size);
  return true;
}

template <typename T>
void GatherV2CPUKernel<T>::CheckParam(const CNodePtr &kernel_node) {
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
