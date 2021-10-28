/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "backend/kernel_compiler/cpu/shift_cpu_kernel.h"
#include "common/thread_pool.h"

namespace mindspore {
namespace kernel {
template <typename T>
void ShiftCpuKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  size_t input_count = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_count != 2) {
    MS_LOG(EXCEPTION) << input_count << " inputs were provided, but Shift expects 2.";
  }

  size_t output_count = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_count != 1) {
    MS_LOG(EXCEPTION) << "Number of outputs is " << output_count << ", but should be 1 for Shift.";
  }

  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, kInputIndex);

  periods_ = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, PERIODS);
  auto axis = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, AXIS);
  size_t axis_t = axis < 0 ? LongToSize(axis + SizeToLong(input_shape.size())) : LongToSize(axis);
  if (axis_t >= input_shape.size()) {
    MS_LOG(EXCEPTION) << "the evaluated axis should be smaller than the dimension of input tensor "
                      << input_shape.size() << "D, but got " << axis_t;
  }

  axisIterator_.Init(input_shape, axis_t);

  // index calculation
  if (periods_ > 0) {
    fill_begin_ = 0;
    fill_size_ = periods_;

    copy_src_begin_ = 0;
    copy_dst_begin_ = periods_;
    copy_size_ = SizeToLong(input_shape[axis]) - periods_;
  } else if (periods_ < 0) {
    fill_begin_ = SizeToLong(input_shape[axis]) + periods_;
    fill_size_ = -periods_;

    copy_src_begin_ = -periods_;
    copy_dst_begin_ = 0;
    copy_size_ = SizeToLong(input_shape[axis]) + periods_;
  }
}

template <typename T>
bool ShiftCpuKernel<T>::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                               const std::vector<AddressPtr> &outputs) {
  if (inputs.size() != 2 || outputs.size() != 1) {
    MS_LOG(EXCEPTION) << "Sort needs 2 input and 1 outputs, but get inputs: " << inputs.size()
                      << "outputs: " << outputs.size();
  }
  auto input = reinterpret_cast<T *>(inputs[0]->addr);
  const auto fill_value = reinterpret_cast<T *>(inputs[1]->addr)[0];
  auto output = reinterpret_cast<T *>(outputs[0]->addr);

  if (outputs[0]->size != inputs[0]->size) {
    MS_LOG(EXCEPTION) << "Error output data size!";
  }

  // if periods_ is 0, do nothing
  if (periods_ == 0) {
    // directly copy input to output
    auto ret = memcpy_s(output, outputs[0]->size, input, inputs[0]->size);
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "memcpy_s failed";
    }
    return true;
  }

  const int64_t outer_size = SizeToLong(axisIterator_.OuterSize());
  const int64_t axis_size = SizeToLong(axisIterator_.AxisSize());
  const int64_t inner_size = SizeToLong(axisIterator_.InnerSize());

  // periods is larger than size, all value of the tensor would be fill_value
  if (std::abs(periods_) >= axis_size) {
    (void)std::fill_n(output, outer_size * axis_size * inner_size, fill_value);
    return true;
  }

  if (inputs[0]->size != outer_size * axis_size * inner_size * sizeof(T)) {
    MS_LOG(EXCEPTION) << "Error input data size!";
  }

  // check if the tensor is linear
  if ((inner_size == 1) && (outer_size == 1)) {
    // treat it as a simple 1D array
    size_t copy_size = copy_size_ * sizeof(T);
    size_t dst_max_size = outputs[0]->size - copy_dst_begin_;
    auto ret = memcpy_s(output + copy_dst_begin_, dst_max_size, input + copy_src_begin_, copy_size);
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "memcpy_s failed";
    }
    (void)std::fill_n(output + fill_begin_, fill_size_, fill_value);
    return true;
  }

  // normal procedure
  std::vector<common::Task> tasks;
  tasks.reserve(outer_size);
  for (int i = 0; i < outer_size; ++i) {
    (void)tasks.emplace_back([this, i, fill_value, axis_size, inner_size, input, output, outputs] {
      size_t offset = i * axis_size * inner_size;
      size_t input_offset = offset + copy_src_begin_ * inner_size;
      size_t output_offset = offset + copy_dst_begin_ * inner_size;
      size_t copy_size = copy_size_ * inner_size * sizeof(T);
      size_t dst_max_size = outputs[0]->size - output_offset;
      auto ret = memcpy_s(output + output_offset, dst_max_size, input + input_offset, copy_size);
      if (ret != EOK) {
        MS_LOG(EXCEPTION) << "memcpy_s failed";
      }
      size_t fill_offset = offset + fill_begin_ * inner_size;
      (void)std::fill_n(output + fill_offset, fill_size_ * inner_size, fill_value);
      return common::SUCCESS;
    });
  }
  (void)common::ThreadPool::GetInstance().SyncRun(tasks);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
