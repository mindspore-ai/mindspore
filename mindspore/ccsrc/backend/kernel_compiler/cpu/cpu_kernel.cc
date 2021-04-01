/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include <algorithm>
#include <utility>
#include "common/thread_pool.h"

namespace mindspore {
namespace kernel {
void CPUKernel::InitInputOutputSize(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  for (size_t input_index = 0; input_index < input_num; ++input_index) {
    TypeId type_id = AnfAlgo::GetInputDeviceDataType(kernel_node, input_index);
    size_t type_size = GetTypeByte(TypeIdToType(type_id));
    std::vector<size_t> shape = AnfAlgo::GetInputDeviceShape(kernel_node, input_index);
    size_t tensor_size =
      shape.empty() ? type_size : std::accumulate(shape.begin(), shape.end(), type_size, std::multiplies<size_t>());
    input_size_list_.emplace_back(tensor_size);
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  for (size_t output_index = 0; output_index < output_num; ++output_index) {
    TypeId type_id = AnfAlgo::GetOutputDeviceDataType(kernel_node, output_index);
    size_t type_size = GetTypeByte(TypeIdToType(type_id));
    std::vector<size_t> shape = AnfAlgo::GetOutputDeviceShape(kernel_node, output_index);
    size_t tensor_size =
      shape.empty() ? type_size : std::accumulate(shape.begin(), shape.end(), type_size, std::multiplies<size_t>());
    output_size_list_.emplace_back(tensor_size);
  }
}

void CPUKernel::Init(const CNodePtr &kernel_node) {
  InitKernel(kernel_node);
  InitInputOutputSize(kernel_node);
}

void CPUKernelUtils::ExpandDimsTo4(std::vector<size_t> *shape) {
  auto len = shape->size();
  if (len < 4) {
    for (size_t i = 0; i < 4 - len; ++i) {
      shape->insert(shape->begin(), 1);
    }
  }
}

size_t CPUKernelUtils::CalcOffset(const std::vector<size_t> &shape, size_t dim0, size_t dim1, size_t dim2,
                                  size_t dim3) {
  size_t offset = dim0 * shape[1] * shape[2] * shape[3] + dim1 * shape[2] * shape[3] + dim2 * shape[3] + dim3;
  return offset;
}

size_t CPUKernelUtils::GetElementNumOnAxis(const std::vector<size_t> &shape, int axis) {
  if (axis < 0) {
    axis = axis + SizeToInt(shape.size());
  }
  size_t result = 1;
  for (int j = 3; j > axis; --j) {
    result *= shape[j];
  }
  return result;
}

void CPUKernelUtils::GetElementNumEveryDim(const std::vector<size_t> &shape, std::vector<size_t> *element_num) {
  size_t accumulation = 1;
  element_num->emplace_back(1);
  for (size_t i = shape.size() - 1; i > 0; --i) {
    accumulation *= shape[i];
    element_num->emplace_back(accumulation);
  }
  std::reverse(element_num->begin(), element_num->end());
}

void CPUKernelUtils::ParallelFor(const CTask &task, size_t count) {
  auto max_thread_num = common::ThreadPool::GetInstance().GetSyncRunThreadNum();
  const float block_size = 128.0;
  size_t thread_num = count < block_size * max_thread_num ? std::ceil(count / block_size) : max_thread_num;
  std::vector<common::Task> tasks;
  size_t start = 0;
  size_t once_compute_size = (count + thread_num - 1) / thread_num;
  while (start < count) {
    size_t end = (start + once_compute_size) > count ? count : (start + once_compute_size);
    auto block = [&, start, end]() {
      task(start, end);
      return common::SUCCESS;
    };
    tasks.emplace_back(block);
    start += once_compute_size;
  }
  common::ThreadPool::GetInstance().SyncRun(tasks);
}

std::vector<size_t> CPUKernelUtils::FlatShapeByAxis(const std::vector<size_t> &shape, int axis) {
  if (axis < 0) {
    axis = axis + SizeToInt(shape.size());
  }
  size_t dim_row = 1;
  size_t dim_col = 1;
  std::vector<size_t> flat_shape;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (SizeToInt(i) < axis) {
      dim_row *= shape[i];
    } else {
      dim_col *= shape[i];
    }
  }
  flat_shape.push_back(dim_row);
  flat_shape.push_back(dim_col);
  return flat_shape;
}

BroadcastIterator::BroadcastIterator(std::vector<size_t> input_shape_a, std::vector<size_t> input_shape_b,
                                     std::vector<size_t> output_shape)
    : input_shape_a_(std::move(input_shape_a)),
      input_shape_b_(std::move(input_shape_b)),
      output_shape_(std::move(output_shape)) {
  output_dimension_ = SizeToInt(output_shape_.size());  // Assign dimension to int for iterator
  BroadcastShape();
  // Allocate strides memory
  input_strides_a_.resize(output_dimension_);
  input_strides_b_.resize(output_dimension_);
  input_back_strides_a_.resize(output_dimension_);
  input_back_strides_b_.resize(output_dimension_);
  coordinates_.resize(output_dimension_);
  InitStrides();
}

void BroadcastIterator::SetPos(size_t pos) {
  for (int i = output_dimension_ - 1; i >= 0 && pos != 0; --i) {
    coordinates_[i] = pos % output_shape_[i];
    input_pos_[0] += coordinates_[i] * input_strides_a_[i];
    input_pos_[1] += coordinates_[i] * input_strides_b_[i];
    pos /= output_shape_[i];
  }
}

void BroadcastIterator::GenNextPos() {
  // Calculate output next coordinate
  for (int i = output_dimension_ - 1; i >= 0; --i) {
    if (coordinates_[i] + 1 == output_shape_[i]) {
      coordinates_[i] = 0;
      input_pos_[0] -= input_back_strides_a_[i];
      input_pos_[1] -= input_back_strides_b_[i];
    } else {
      ++coordinates_[i];
      input_pos_[0] += input_strides_a_[i];
      input_pos_[1] += input_strides_b_[i];
      break;
    }
  }
}

void BroadcastIterator::BroadcastShape() {
  int input_dimension_a = input_shape_a_.size();
  if (input_dimension_a < output_dimension_) {
    input_shape_a_.insert(input_shape_a_.begin(), output_dimension_ - input_dimension_a, 1);
  }

  int input_dimension_b = input_shape_b_.size();
  if (input_dimension_b < output_dimension_) {
    input_shape_b_.insert(input_shape_b_.begin(), output_dimension_ - input_dimension_b, 1);
  }
}

void BroadcastIterator::InitStrides() {
  input_strides_a_[output_dimension_ - 1] = 1;
  input_strides_b_[output_dimension_ - 1] = 1;
  for (int i = output_dimension_ - 2; i >= 0; --i) {
    input_strides_a_[i] = input_shape_a_[i + 1] * input_strides_a_[i + 1];
    input_strides_b_[i] = input_shape_b_[i + 1] * input_strides_b_[i + 1];
    input_back_strides_a_[i + 1] = (input_shape_a_[i + 1] - 1) * input_strides_a_[i + 1];
    input_back_strides_b_[i + 1] = (input_shape_b_[i + 1] - 1) * input_strides_b_[i + 1];
  }

  // Update strides for broadcast
  // While the axis value is 1, the stride is 0
  std::transform(input_strides_a_.begin(), input_strides_a_.end(), input_shape_a_.begin(), input_strides_a_.begin(),
                 [](const auto &a, const auto &b) { return b == 1 ? 0 : a; });
  std::transform(input_strides_b_.begin(), input_strides_b_.end(), input_shape_b_.begin(), input_strides_b_.begin(),
                 [](const auto &a, const auto &b) { return b == 1 ? 0 : a; });
}

TransposeIterator::TransposeIterator(std::vector<size_t> output_shape, std::vector<size_t> axes,
                                     const std::vector<size_t> &input_shape)
    : shape_(std::move(output_shape)), axes_(std::move(axes)) {
  // Calculate strides
  dimension_ = shape_.size();
  std::vector<uint32_t> strides(dimension_, 1);
  for (int i = dimension_ - 2; i >= 0; --i) {
    strides[i] = input_shape[i + 1] * strides[i + 1];
  }

  // Swap shape ans strides and calculate back strides
  strides_.resize(dimension_);
  back_strides_.resize(dimension_);
  for (int i = dimension_ - 1; i >= 0; --i) {
    strides_[i] = strides[axes_[i]];
    back_strides_[i] = (shape_[i] - 1) * strides_[i];
  }

  // Calculate coordinate by pos
  coordinates_.resize(dimension_);
}

void TransposeIterator::SetPos(size_t pos) {
  for (int i = dimension_ - 1; i >= 0 && pos != 0; --i) {
    coordinates_[i] = pos % shape_[i];
    pos_ += coordinates_[i] * strides_[i];
    pos /= shape_[i];
  }
}

void TransposeIterator::GenNextPos() {
  for (int i = dimension_ - 1; i >= 0; --i) {
    if (coordinates_[i] + 1 == shape_[i]) {
      coordinates_[i] = 0;
      pos_ -= back_strides_[i];
    } else {
      coordinates_[i]++;
      pos_ += strides_[i];
      break;
    }
  }
}
}  // namespace kernel
}  // namespace mindspore
