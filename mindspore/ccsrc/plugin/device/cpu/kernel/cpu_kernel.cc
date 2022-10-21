/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/cpu_kernel.h"

#include <algorithm>
#include <utility>
#include <cmath>
#include <map>
#include <set>
#include <numeric>
#include "kernel/oplib/oplib.h"
#include "utils/profile.h"
#include "runtime/graph_scheduler/actor/actor_common.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
std::vector<KernelAttr> NativeCpuKernelMod::GetAllSupportedList(const std::string &kernel_name) {
  auto iter = support_map_.find(kernel_name);
  if (iter == support_map_.end()) {
    std::vector<KernelAttr> kernel_attrs;
    auto kernel_support = GetOpSupport();
    (void)kernel_attrs.insert(kernel_attrs.end(), kernel_support.begin(), kernel_support.end());
    if (!kernel_attrs.empty() && kernel_attrs[0].GetSkipCheck()) {
      (void)support_map_.emplace(kernel_name, kernel_attrs);
      return support_map_[kernel_name];
    }
    if (kernel_attrs.empty()) {
      auto oplib_support = GetSupportFromOpLib(kernel_name);
      (void)kernel_attrs.insert(kernel_attrs.end(), oplib_support.begin(), oplib_support.end());
    }
    (void)support_map_.emplace(kernel_name, kernel_attrs);
  }

  return support_map_[kernel_name];
}

std::vector<KernelAttr> NativeCpuKernelMod::GetSupportFromOpLib(const std::string &kernel_name) const {
  static std::set<std::string> same_op_name = {"Concat", "Pack", "Stack",        "Split",        "Transpose",
                                               "Unpack", "AddN", "ConcatOffset", "DynamicStitch"};
  std::vector<KernelAttr> support_kernel_attrs;
  auto op_info = mindspore::kernel::OpLib::FindOp(kernel_name, kernel::OpImplyType::kImplyCPU);
  if (op_info == nullptr) {
    MS_LOG(WARNING) << "Not find op[" << kernel_name << "] in cpu. For more details, "
                    << "please refer to the list of supported cpu operations at https://www.mindspore.cn.";
    return support_kernel_attrs;
  }

  auto inputs_ptr = op_info->inputs_ptr();
  auto outputs_ptr = op_info->outputs_ptr();
  if (outputs_ptr.empty()) {
    MS_LOG(WARNING) << "The output dimension of operator '" << kernel_name << "' can not be zero.";
    return support_kernel_attrs;
  }

  auto support_size = outputs_ptr[0]->dtypes().size();
  for (size_t i = 0; i < support_size; i++) {
    KernelAttr kernel_attr;
    for (size_t j = 0; j < inputs_ptr.size(); j++) {
      auto input_dtypes = inputs_ptr[j]->dtypes();
      auto input_formats = inputs_ptr[j]->formats();
      (void)kernel_attr.AddInputAttr(kernel::DtypeToTypeId(input_dtypes[i]), input_formats[i]);
    }
    for (size_t j = 0; j < outputs_ptr.size(); j++) {
      auto output_dtypes = outputs_ptr[j]->dtypes();
      auto output_formats = outputs_ptr[j]->formats();
      (void)kernel_attr.AddOutputAttr(kernel::DtypeToTypeId(output_dtypes[i]), output_formats[i]);
    }
    if (same_op_name.count(op_info->op_name()) != 0) {
      (void)kernel_attr.AddAllSameAttr(true);
    }
    support_kernel_attrs.push_back(kernel_attr);
  }

  return support_kernel_attrs;
}

int DeprecatedNativeCpuKernelMod::Resize(const BaseOperatorPtr &, const std::vector<KernelTensorPtr> &,
                                         const std::vector<KernelTensorPtr> &,
                                         const std::map<uint32_t, tensor::TensorPtr> &) {
  auto cnode = cnode_ptr_.lock();
  MS_EXCEPTION_IF_NULL(cnode);
  MS_LOG(DEBUG) << "Update Args: " << cnode->fullname_with_scope();

  Init(cnode);
  return 0;
}

void DeprecatedNativeCpuKernelMod::InitInputOutputSize(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  input_size_list_.clear();
  output_size_list_.clear();
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  for (size_t input_index = 0; input_index < input_num; ++input_index) {
    TypeId type_id = AnfAlgo::GetInputDeviceDataType(kernel_node, input_index);
    size_t type_size = GetTypeByte(TypeIdToType(type_id));
    auto shape = AnfAlgo::GetInputDeviceShape(kernel_node, input_index);
    size_t tensor_size =
      shape.empty() ? type_size : std::accumulate(shape.begin(), shape.end(), type_size, std::multiplies<size_t>());
    tensor_size = std::max(tensor_size, type_size);
    (void)input_size_list_.emplace_back(tensor_size);
  }
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  for (size_t output_index = 0; output_index < output_num; ++output_index) {
    size_t tensor_size = AnfAlgo::GetOutputTensorMemSize(kernel_node, output_index);
    (void)output_size_list_.emplace_back(tensor_size);
  }
}

void DeprecatedNativeCpuKernelMod::Init(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  if (cnode_ptr_.lock() == nullptr) {
    cnode_ptr_ = kernel_node;
  }

  workspace_size_list_.clear();
  InitKernel(kernel_node);
  InitInputOutputSize(kernel_node);
}

std::vector<TypeId> DeprecatedNativeCpuKernelMod::GetInputDtypes(const CNodePtr &kernel_node) const {
  std::vector<TypeId> input_types;
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  for (size_t input_index = 0; input_index < input_num; ++input_index) {
    auto dtype = AnfAlgo::GetPrevNodeOutputDeviceDataType(kernel_node, input_index);
    (void)input_types.emplace_back(dtype);
  }
  return input_types;
}

std::vector<TypeId> DeprecatedNativeCpuKernelMod::GetOutputDtypes(const CNodePtr &kernel_node) const {
  std::vector<TypeId> output_types;
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  for (size_t output_index = 0; output_index < output_num; ++output_index) {
    auto dtype = common::AnfAlgo::GetOutputInferDataType(kernel_node, output_index);
    (void)output_types.emplace_back(dtype);
  }
  return output_types;
}

void DeprecatedNativeCpuKernelMod::SetCpuRefMapToKernelInfo(const CNodePtr &apply_kernel) {
  kernel::SetCpuRefMapToKernelInfo(apply_kernel, GetOpSupport());
}

void CPUKernelUtils::ExpandDimsTo4(ShapeVector *shape) {
  MS_EXCEPTION_IF_NULL(shape);
  auto len = shape->size();
  const size_t expect_dims = 4;
  if (len < expect_dims) {
    for (size_t i = 0; i < expect_dims - len; ++i) {
      (void)shape->insert(shape->begin(), 1);
    }
  }
}

size_t CPUKernelUtils::CalcOffset(const ShapeVector &shape, size_t dim0, size_t dim1, size_t dim2, size_t dim3) {
  size_t offset = dim0 * LongToSize(shape[1]) * LongToSize(shape[2]) * LongToSize(shape[3]) +
                  dim1 * LongToSize(shape[2]) * LongToSize(shape[3]) + dim2 * LongToSize(shape[3]) + dim3;
  return offset;
}

size_t CPUKernelUtils::GetElementNumOnAxis(const ShapeVector &shape, int axis) {
  if (axis < 0) {
    axis = axis + SizeToInt(shape.size());
  }
  int64_t result = 1;
  for (int j = 3; j > axis; --j) {
    result *= shape[j];
  }
  return LongToSize(result);
}

void CPUKernelUtils::GetElementNumEveryDim(const ShapeVector &shape, std::vector<size_t> *element_num) {
  size_t accumulation = 1;
  MS_EXCEPTION_IF_NULL(element_num);
  (void)element_num->emplace_back(1);
  for (size_t i = shape.size() - 1; i > 0; --i) {
    accumulation *= LongToSizeClipNeg(shape[i]);
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
  ParallelLaunch(tasks);
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
    size_t actor_thread_num = 0;
    size_t actor_and_kernel_thread_num = 0;
    runtime::ComputeThreadNums(&actor_thread_num, &actor_and_kernel_thread_num);
    size_t actor_queue_size = 81920;
    (void)actor_manager->Initialize(true, actor_thread_num, actor_and_kernel_thread_num, actor_queue_size);
    thread_pool = actor_manager->GetActorThreadPool();
    MS_EXCEPTION_IF_NULL(thread_pool);
  }
  thread_pool->SetKernelThreadMaxSpinCount(kDefaultKernelSpinCount);
  return thread_pool;
}

// Use threadpool of mindrt
void ParallelLaunch(const CTask &task, size_t count, float block_size, Content content, ThreadPool *pool) {
  if (count == 0) {
    return;
  }
  auto thread_pool = pool == nullptr ? GetActorMgrInnerThreadPool() : pool;
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

void ParallelLaunch(const std::vector<common::Task> &tasks, Content content, ThreadPool *pool) {
  size_t count = tasks.size();
  if (count == 0) {
    return;
  }
  auto thread_pool = pool == nullptr ? GetActorMgrInnerThreadPool() : pool;
  size_t kernel_thread_num = thread_pool->GetKernelThreadNum();
  if (kernel_thread_num == 0) {
    MS_LOG(EXCEPTION) << "Actor inner pool has been init, but kernel thread is 0!";
  }

  size_t thread_num = count < kernel_thread_num ? count : kernel_thread_num;
  size_t once_compute_size = (count + thread_num - 1) / thread_num;
  size_t task_num = count / once_compute_size;
  if (count % once_compute_size != 0) {
    task_num += 1;
  }
  auto func = [&](void *, int task_id, float, float) {
    size_t start = task_id * once_compute_size;
    size_t end = (start + once_compute_size) > count ? count : (start + once_compute_size);
    for (size_t i = start; i < end; ++i) {
      (void)tasks[i]();
    }
    return common::SUCCESS;
  };
  (void)thread_pool->ParallelLaunch(func, content, task_num);
}

void ParallelLaunchAutoSearch(const CTask &task, size_t count, Content content,
                              ParallelSearchInfo *parallel_search_info, ThreadPool *pool) {
  if (!parallel_search_info->kernel_thread_num_set) {
    auto thread_pool = pool == nullptr ? GetActorMgrInnerThreadPool() : pool;
    size_t kernel_thread_num = thread_pool->GetKernelThreadNum();
    if (kernel_thread_num == 0) {
      MS_LOG(EXCEPTION) << "Actor inner pool has been init, but kernel thread is 0!";
    }
    size_t max_pow_current = parallel_search_info->max_pow - 1;
    while (std::pow(2.0f, max_pow_current) <= static_cast<float>(kernel_thread_num)) {
      max_pow_current++;
    }
    parallel_search_info->max_pow = max_pow_current + 1;
    parallel_search_info->kernel_thread_num_set = true;
  }
  const size_t AVG_COUNT = 5;
  size_t current_pow = parallel_search_info->search_count / AVG_COUNT;
  if (current_pow < parallel_search_info->max_pow) {
    if (parallel_search_info->search_count % AVG_COUNT == 0) {
      parallel_search_info->tmp_sum_cost_time = 0;
    }
    float block_size = static_cast<float>(count) / std::pow(2.0f, current_pow);
    double start_time = GetTime();
    ParallelLaunch(task, count, block_size, content, pool);
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
        parallel_search_info->search_count = AVG_COUNT * parallel_search_info->max_pow;
      }
    }
  } else {
    ParallelLaunch(task, count, parallel_search_info->best_block_size, content, pool);
  }
}

ShapeVector CPUKernelUtils::FlatShapeByAxis(const ShapeVector &shape, int axis) {
  if (axis < 0) {
    axis = axis + SizeToInt(shape.size());
  }
  int64_t dim_row = 1;
  int64_t dim_col = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (SizeToInt(i) < axis) {
      dim_row *= shape[i];
    } else {
      dim_col *= shape[i];
    }
  }
  // referred to Copy elision https://en.cppreference.com/w/cpp/language/copy_elision
  // returning a vector won't cause extra vector constructed or moved
  return ShapeVector{dim_row, dim_col};
}

BroadcastIterator::BroadcastIterator(ShapeVector input_shape_a, ShapeVector input_shape_b, ShapeVector output_shape)
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
  if (output_dimension_ <= 0) {
    return;
  }
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

MultipleBroadcastIterator::MultipleBroadcastIterator(std::vector<shape_info> multi_inputs, shape_info output_shape)
    : multi_inputs_(std::move(multi_inputs)), output_shape_(std::move(output_shape)) {
  output_dimension_ = SizeToInt(output_shape_.size());
  // Assign dimension to int for iterator
  BroadcastShape();
  input_pos_.resize(multi_inputs_.size());
  // Allocate strides memory
  multi_inputs_strides_.resize(multi_inputs_.size(), std::vector<int64_t>(output_dimension_, 0));
  multi_inputs_back_strides_.resize(multi_inputs_.size(), std::vector<int64_t>(output_dimension_, 0));
  coordinates_.resize(output_dimension_);
  InitStrides();
}

void MultipleBroadcastIterator::SetPos(size_t pos) {
  for (int i = output_dimension_ - 1; i >= 0 && pos != 0; --i) {
    coordinates_[i] = pos % output_shape_[i];
    for (size_t j = 0; j < input_pos_.size(); ++j) {
      input_pos_[j] += coordinates_[i] * multi_inputs_strides_[j][i];
    }
    pos /= output_shape_[i];
  }
}

void MultipleBroadcastIterator::GenNextPos() {
  // Calculate output next coordinate
  for (int i = output_dimension_ - 1; i >= 0; --i) {
    if (coordinates_[i] + 1 == output_shape_[i]) {
      coordinates_[i] = 0;
      for (size_t j = 0; j < input_pos_.size(); ++j) {
        input_pos_[j] -= multi_inputs_back_strides_[j][i];
      }
    } else {
      ++coordinates_[i];
      for (size_t j = 0; j < input_pos_.size(); ++j) {
        input_pos_[j] += multi_inputs_strides_[j][i];
      }
      break;
    }
  }
}

void MultipleBroadcastIterator::BroadcastShape() {
  for (auto &multi_input : multi_inputs_) {
    int input_dimension = SizeToInt(multi_input.size());
    if (input_dimension < output_dimension_) {
      (void)multi_input.insert(multi_input.begin(), IntToSize(output_dimension_ - input_dimension), 1);
    }
  }
}

void MultipleBroadcastIterator::InitStrides() {
  for (size_t i = 0; i < multi_inputs_.size(); ++i) {
    multi_inputs_strides_[i][output_dimension_ - 1] = 1;
    for (int j = output_dimension_ - 2; j >= 0; --j) {
      multi_inputs_strides_[i][j] = multi_inputs_[i][j + 1] * multi_inputs_strides_[i][j + 1];
      multi_inputs_back_strides_[i][j + 1] = (multi_inputs_[i][j + 1] - 1) * multi_inputs_strides_[i][j + 1];
    }
    // Update strides for broadcast
    // While the axis value is 1, the stride is 0
    (void)std::transform(multi_inputs_strides_[i].begin(), multi_inputs_strides_[i].end(), multi_inputs_[i].begin(),
                         multi_inputs_strides_[i].begin(), [](const auto &a, const auto &b) { return b == 1 ? 0 : a; });
  }
}

TransposeIterator::TransposeIterator(ShapeVector output_shape, std::vector<size_t> axes, const ShapeVector &input_shape)
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

ShapeVector CPUKernelUtils::GetBroadcastShape(const ShapeVector &x, const ShapeVector &y) {
  size_t x_len = x.size();
  size_t y_len = y.size();
  size_t length = x_len < y_len ? x_len : y_len;
  ShapeVector broadcast_shape;
  ShapeVector broadcast_shape_back;
  for (int i = -SizeToInt(length); i < 0; ++i) {
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

void AxisIterator::Init(const ShapeVector &input_shape, size_t axis) {
  outer_size_ = 1;
  for (size_t i = 0; i < axis; i++) {
    outer_size_ *= LongToSize(input_shape[i]);
  }

  axis_size_ = LongToSize(input_shape[axis]);

  inner_size_ = 1;
  for (size_t i = axis + 1; i < input_shape.size(); ++i) {
    inner_size_ *= LongToSize(input_shape[i]);
  }
}

int Sign(float x) {
  if (x > 0) {
    return 1;
  }
  if (x < 0) {
    return -1;
  }
  return 0;
}
}  // namespace kernel
}  // namespace mindspore
