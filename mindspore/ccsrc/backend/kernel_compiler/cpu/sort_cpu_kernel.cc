/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "backend/kernel_compiler/cpu/sort_cpu_kernel.h"
#include <vector>
#include "common/thread_pool.h"

namespace mindspore {
namespace kernel {

template <typename T>
void SortCpuKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  size_t input_count = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_count != 1) {
    MS_LOG(EXCEPTION) << input_count << " inputs were provided, but Sort expects 1.";
  }

  size_t output_count = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_count != 2) {
    MS_LOG(EXCEPTION) << "Number of outputs is " << output_count << ", but should be 2 for Sort.";
  }

  auto x_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  auto input_rank = x_shape_.size();

  descending_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, "descending");
  auto axis = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "axis");
  if (axis < 0) {
    axis += input_rank;
  }

  if ((axis < 0) || (axis >= static_cast<int64_t>(input_rank))) {
    MS_LOG(EXCEPTION) << "evaluated axis is " << axis << ", but should be in range [0, " << input_rank << "].";
  }

  size_t axis_t = axis;

  outer_size_ = 1;
  for (size_t i = 0; i < axis_t; i++) {
    outer_size_ *= x_shape_[i];
  }

  axis_size_ = x_shape_[axis_t];

  inner_size_ = 1;
  for (size_t i = axis_t + 1; i < input_rank; ++i) {
    inner_size_ *= x_shape_[i];
  }
}

template <typename T>
bool SortCpuKernel<T>::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                              const std::vector<AddressPtr> &outputs) {
  if (inputs.size() != 1 || outputs.size() != 2) {
    MS_LOG(EXCEPTION) << "Sort needs 1 input and 2 outputs, but get inputs: " << inputs.size()
                      << "outputs: " << outputs.size();
  }
  if (inputs[0]->size != outer_size_ * axis_size_ * inner_size_ * sizeof(T)) {
    MS_LOG(EXCEPTION) << "Error input data size!";
  }
  auto input = reinterpret_cast<T *>(inputs[0]->addr);
  auto output = reinterpret_cast<T *>(outputs[0]->addr);
  auto indices = reinterpret_cast<int *>(outputs[1]->addr);

  if (outputs[0]->size != inputs[0]->size) {
    MS_LOG(EXCEPTION) << "Error output data size!";
  }

  std::function<bool(size_t, size_t)> comparator;
  if (descending_) {
    comparator = [&input](size_t index_1, size_t index_2) { return input[index_1] > input[index_2]; };
  } else {
    comparator = [&input](size_t index_1, size_t index_2) { return input[index_1] < input[index_2]; };
  }

  std::vector<common::Task> tasks;
  tasks.reserve(outer_size_ * inner_size_);
  for (size_t i = 0; i < outer_size_; ++i) {
    const auto out_offset = i * axis_size_ * inner_size_;
    for (size_t j = 0; j < inner_size_; ++j) {
      const auto axis_offset = out_offset + j;
      auto task = [this, axis_offset, &input, &indices, &output, &comparator]() {
        std::vector<size_t> idx(axis_size_);
        // fill idx starts with out_offset + j, step inner_size_
        for (size_t k = 0; k < axis_size_; ++k) {
          idx[k] = axis_offset + k * inner_size_;
        }

        std::stable_sort(idx.begin(), idx.end(), comparator);

        for (size_t k = 0; k < axis_size_; ++k) {
          const auto index = axis_offset + k * inner_size_;
          indices[index] = (SizeToInt(idx[k]) - SizeToInt(axis_offset)) / inner_size_;
          output[index] = input[idx[k]];
        }
        return common::SUCCESS;
      };
      tasks.emplace_back(task);
    }
  }
  common::ThreadPool::GetInstance().SyncRun(tasks);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
