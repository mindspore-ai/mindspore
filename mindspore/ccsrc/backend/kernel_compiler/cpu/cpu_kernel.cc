/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include <cmath>

#include "common/thread_pool.h"
#include "utils/profile.h"

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
    tensor_size = std::max(tensor_size, type_size);
    (void)input_size_list_.emplace_back(tensor_size);
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  for (size_t output_index = 0; output_index < output_num; ++output_index) {
    TypeId type_id = AnfAlgo::GetOutputDeviceDataType(kernel_node, output_index);
    size_t type_size = GetTypeByte(TypeIdToType(type_id));
    std::vector<size_t> shape = AnfAlgo::GetOutputDeviceShape(kernel_node, output_index);
    size_t tensor_size =
      shape.empty() ? type_size : std::accumulate(shape.begin(), shape.end(), type_size, std::multiplies<size_t>());
    tensor_size = std::max(tensor_size, type_size);
    (void)output_size_list_.emplace_back(tensor_size);
  }
}

void CPUKernel::Init(const CNodePtr &kernel_node) {
  InitKernel(kernel_node);
  InitInputOutputSize(kernel_node);
}

void CPUKernelUtils::ExpandDimsTo4(std::vector<size_t> *shape) {
  MS_EXCEPTION_IF_NULL(shape);
  auto len = shape->size();
  if (len < 4) {
    for (size_t i = 0; i < 4 - len; ++i) {
      (void)shape->insert(shape->begin(), 1);
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
  MS_EXCEPTION_IF_NULL(element_num);
  (void)element_num->emplace_back(1);
  for (size_t i = shape.size() - 1; i > 0; --i) {
    accumulation *= shape[i];
    (void)element_num->emplace_back(accumulation);
  }
  std::reverse(element_num->begin(), element_num->end());
}

void CPUKernelUtils::ParallelFor(const CTask &task, size_t count, float block_size) {
  auto max_thread_num = common::ThreadPool::GetInstance().GetSyncRunThreadNum();
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
    (void)tasks.emplace_back(block);
    start += once_compute_size;
  }
  (void)common::ThreadPool::GetInstance().SyncRun(tasks);
}

// Search for best block_size to get best thread num : 1 2 4 8 16 23(32)
// Each block_size runs 5 times to get an average cpu kernel cost time.
// If the speed of block_size[i] is slower than block_size[i-2], than we
// assume that  block_size[i-2] is the best block_size.
void CPUKernelUtils::ParallelForAutoSearch(const CTask &task, size_t count, ParallelSearchInfo *parallel_search_info) {
  const size_t MAX_POW = 6;
  const size_t AVG_COUNT = 5;
  MS_EXCEPTION_IF_NULL(parallel_search_info);
  size_t current_pow = parallel_search_info->search_count / AVG_COUNT;
  if (current_pow < MAX_POW) {
    if (parallel_search_info->search_count % AVG_COUNT == 0) {
      parallel_search_info->tmp_sum_cost_time = 0;
    }
    float block_size = static_cast<float>(count) / std::pow(2.0f, current_pow);
    double start_time = GetTime();
    ParallelFor(task, count, block_size);
    double cost_time = GetTime() - start_time;
    parallel_search_info->tmp_sum_cost_time += cost_time;
    parallel_search_info->search_count++;
    if (parallel_search_info->search_count % AVG_COUNT == 0) {
      double avg_time = parallel_search_info->tmp_sum_cost_time / AVG_COUNT;
      if (parallel_search_info->min_cost_time > avg_time) {
        parallel_search_info->min_cost_time = avg_time;
        parallel_search_info->best_block_size = block_size;
        parallel_search_info->best_pow = current_pow;
      } else if (current_pow - parallel_search_info->best_pow >= 2) {
        parallel_search_info->search_count = AVG_COUNT * MAX_POW;
      }
    }
  } else {
    ParallelFor(task, count, parallel_search_info->best_block_size);
  }
}

ActorThreadPool *GetActorMgrInnerThreadPool() {
  auto actor_manager = ActorMgr::GetActorMgrRef();
  MS_EXCEPTION_IF_NULL(actor_manager);
  auto thread_pool = actor_manager->GetActorThreadPool();
  // Init thread_pool if env is windows or ascend, in case that it won't be init in graph_scheduler.
  if (thread_pool == nullptr) {
    const size_t kMaxThreadNum = 23;
    size_t max_thread_num = std::thread::hardware_concurrency() - 1;
#if ENABLE_D || ENABLE_GPU
    const size_t kDeviceNum = 8;
    max_thread_num /= kDeviceNum;
#endif
    if (max_thread_num < 1) {
      max_thread_num = 1;
    }
    max_thread_num = max_thread_num < kMaxThreadNum ? max_thread_num : kMaxThreadNum;
    (void)actor_manager->Initialize(true, 0, max_thread_num);
    thread_pool = actor_manager->GetActorThreadPool();
    MS_EXCEPTION_IF_NULL(thread_pool);
  }
  return thread_pool;
}

// Use threadpool of mindrt
void ParallelLaunch(const CTask &task, size_t count, float block_size, Content content) {
  auto thread_pool = GetActorMgrInnerThreadPool();
  size_t kernel_thread_num = thread_pool->GetKernelThreadNum();
  if (kernel_thread_num == 0) {
    MS_LOG(EXCEPTION) << "Actor inner pool has been init, but kernel thread is 0!";
  }

  size_t thread_num = count < block_size * kernel_thread_num ? std::ceil(count / block_size) : kernel_thread_num;
  size_t once_compute_size = (count + thread_num - 1) / thread_num;
  size_t task_num = count / once_compute_size;
  if (count % once_compute_size != 0) {
    task_num += 1;
  }
  auto func = [&](void *, int task_id, float, float) {
    size_t start = task_id * once_compute_size;
    size_t end = (start + once_compute_size) > count ? count : (start + once_compute_size);
    task(start, end);
    return common::SUCCESS;
  };
  (void)thread_pool->ParallelLaunch(func, content, task_num);
}

void ParallelLaunchAutoSearch(const CTask &task, size_t count, Content content,
                              ParallelSearchInfo *parallel_search_info) {
  const size_t MAX_POW = 6;
  const size_t AVG_COUNT = 5;
  size_t current_pow = parallel_search_info->search_count / AVG_COUNT;
  if (current_pow < MAX_POW) {
    if (parallel_search_info->search_count % AVG_COUNT == 0) {
      parallel_search_info->tmp_sum_cost_time = 0;
    }
    float block_size = static_cast<float>(count) / std::pow(2.0f, current_pow);
    double start_time = GetTime();
    ParallelLaunch(task, count, block_size, content);
    double cost_time = GetTime() - start_time;
    parallel_search_info->tmp_sum_cost_time += cost_time;
    parallel_search_info->search_count++;
    if (parallel_search_info->search_count % AVG_COUNT == 0) {
      double avg_time = parallel_search_info->tmp_sum_cost_time / AVG_COUNT;
      if (parallel_search_info->min_cost_time > avg_time) {
        parallel_search_info->min_cost_time = avg_time;
        parallel_search_info->best_block_size = block_size;
        parallel_search_info->best_pow = current_pow;
      } else if (current_pow - parallel_search_info->best_pow >= 2) {
        parallel_search_info->search_count = AVG_COUNT * MAX_POW;
      }
    }
  } else {
    ParallelLaunch(task, count, parallel_search_info->best_block_size, content);
  }
}

std::vector<size_t> CPUKernelUtils::FlatShapeByAxis(const std::vector<size_t> &shape, int axis) {
  if (axis < 0) {
    axis = axis + SizeToInt(shape.size());
  }
  size_t dim_row = 1;
  size_t dim_col = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (SizeToInt(i) < axis) {
      dim_row *= shape[i];
    } else {
      dim_col *= shape[i];
    }
  }
  // referred to Copy elision https://en.cppreference.com/w/cpp/language/copy_elision
  // returning a vector won't cause extra vector constructed or moved
  return std::vector<size_t>{dim_row, dim_col};
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
    (void)input_shape_a_.insert(input_shape_a_.begin(), IntToSize(output_dimension_ - input_dimension_a), 1);
  }

  int input_dimension_b = input_shape_b_.size();
  if (input_dimension_b < output_dimension_) {
    (void)input_shape_b_.insert(input_shape_b_.begin(), IntToSize(output_dimension_ - input_dimension_b), 1);
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
  (void)std::transform(input_strides_a_.begin(), input_strides_a_.end(), input_shape_a_.begin(),
                       input_strides_a_.begin(), [](const auto &a, const auto &b) { return b == 1 ? 0 : a; });
  (void)std::transform(input_strides_b_.begin(), input_strides_b_.end(), input_shape_b_.begin(),
                       input_strides_b_.begin(), [](const auto &a, const auto &b) { return b == 1 ? 0 : a; });
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

std::vector<size_t> CPUKernelUtils::GetBroadcastShape(const std::vector<size_t> &x, const std::vector<size_t> &y) {
  size_t x_len = x.size();
  size_t y_len = y.size();
  size_t length = x_len < y_len ? x_len : y_len;
  std::vector<size_t> broadcast_shape;
  std::vector<size_t> broadcast_shape_back;
  for (int i = -length; i < 0; ++i) {
    if (x[x_len + i] == 1) {
      broadcast_shape_back.push_back(y[y_len + i]);
    } else if (y[y_len + i] == 1) {
      broadcast_shape_back.push_back(x[x_len + i]);
    } else if (x[x_len + i] == y[y_len + i]) {
      broadcast_shape_back.push_back(x[x_len + i]);
    }
  }
  if (length == x_len) {
    for (size_t i = 0; i < y_len - length; ++i) {
      broadcast_shape.push_back(y[i]);
    }
  } else {
    for (size_t i = 0; i < x_len - length; ++i) {
      broadcast_shape.push_back(x[i]);
    }
  }
  for (size_t i = 0; i < length; ++i) {
    broadcast_shape.push_back(broadcast_shape_back[i]);
  }
  return broadcast_shape;
}

void AxisIterator::Init(const std::vector<size_t> &input_shape, size_t axis) {
  outer_size_ = 1;
  for (size_t i = 0; i < axis; i++) {
    outer_size_ *= input_shape[i];
  }

  axis_size_ = input_shape[axis];

  inner_size_ = 1;
  for (size_t i = axis + 1; i < input_shape.size(); ++i) {
    inner_size_ *= input_shape[i];
  }
}
}  // namespace kernel
}  // namespace mindspore
