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

#include "backend/kernel_compiler/cpu/stridedslice_cpu_kernel.h"
#include <utility>
#include <functional>
#include <algorithm>
#include <unordered_map>
#include "common/thread_pool.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kStridedSliceInputsNum = 1;
constexpr size_t kStridedSliceDynamicInputsNum = 4;
constexpr size_t kStridedSliceOutputsNum = 1;
}  // namespace

enum PosType { kBegin, kEnd };

int NormalizePos(int pos, int dim_len, PosType pos_type) {
  if (pos >= 0) {
    int max_pos = pos_type == kBegin ? dim_len - 1 : dim_len;
    return std::min(pos, max_pos);
  }
  int min_pos = pos_type == kBegin ? 0 : -1;
  return std::max(pos + dim_len, min_pos);
}

void StridedSliceCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  cnode_ptr_ = kernel_node;
  input_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  output_shape_ = AnfAlgo::GetOutputInferShape(kernel_node, 0);
  if (input_shape_.size() > DIMENSION_8D || input_shape_.empty()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of 'input_x' should be in range [1D, 8D], but got " << input_shape_.size()
                      << "D.";
  }
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 1) {
    return;
  }
  // for begin, end, stride are const input
  auto begin = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, BEGIN);
  auto end = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, END);
  auto stride = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, STRIDES);
  if (begin.size() != end.size() || begin.size() != stride.size() || begin.size() > input_shape_.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the length of 'begin', 'stride' and 'end' should be equal "
                         "and less than or equal to the dimension of 'input_x', but got the length of 'begin': "
                      << begin.size() << ", the length of 'stride': " << stride.size()
                      << ", the length of 'end': " << end.size()
                      << ", the dimension of 'input_x': " << input_shape_.size();
  }

  InitSliceParam(begin, end, stride);
  parallel_ = MatchParallelPattern();
  if (parallel_) {
    InitParallelParam();
  }
}

bool StridedSliceCPUKernel::MatchParallelPattern() {
  // This function is seeking if that the number of only one dimension
  // is different between input and output. If so, we can do some trick.
  // Example 1:
  // input shape info:  [1, 80, 46, 40]
  // output shape info: [1, 80, 20, 40]
  // Example 2:
  // input shape info:  [1, 46, 40]
  // output shape info: [1, 20, 40]
  if (input_shape_.size() == output_shape_.size()) {
    std::vector<int> axis_list;
    for (size_t i = 0; i < input_shape_.size(); ++i) {
      if (input_shape_[i] != output_shape_[i]) {
        (void)axis_list.emplace_back(i);
      }
    }
    if (axis_list.size() == 1) {
      split_axis_ = axis_list.front();
      return true;
    }
  }
  return false;
}

void StridedSliceCPUKernel::InitParallelParam() {
  outer_ = SizeToInt(
    std::accumulate(input_shape_.begin(), input_shape_.begin() + split_axis_, size_t(1), std::multiplies<size_t>()));
  inner_ = SizeToInt(
    std::accumulate(input_shape_.begin() + split_axis_ + 1, input_shape_.end(), size_t(1), std::multiplies<size_t>()));

  int max_thread_num = SizeToInt(common::ThreadPool::GetInstance().GetSyncRunThreadNum());
  int thread_num = 1;
  if (outer_ == 1) {
    parallel_strategy_ = kOnSplitAxis;
    thread_num = std::min(SizeToInt(output_shape_[split_axis_]), max_thread_num);
    cal_num_per_thread_ = UP_DIV(output_shape_[split_axis_], thread_num);
  } else {
    parallel_strategy_ = kOnOuter;
    thread_num = std::min(outer_, max_thread_num);
    cal_num_per_thread_ = UP_DIV(outer_, thread_num);
  }
  slice_param_.op_parameter_.thread_num_ = thread_num;
}

void StridedSliceCPUKernel::InitSliceParam(const std::vector<int64_t> &begin, const std::vector<int64_t> &end,
                                           const std::vector<int64_t> &stride) {
  static const std::unordered_map<TypeId, std::pair<LiteDataType, int>> type_convert_map = {
    {kNumberTypeBool, {kDataTypeBool, sizeof(bool)}},
    {kNumberTypeInt32, {kDataTypeInt, sizeof(int)}},
    {kNumberTypeFloat32, {kDataTypeFloat, sizeof(float)}},
    {kNumberTypeFloat64, {kDataTypeFloat64, sizeof(double)}}};

  auto type_pair = type_convert_map.find(dtype_);
  if (type_pair == type_convert_map.end()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dtype of 'input_x' should be bool, int32, float32 or float64, but got "
                      << TypeIdToType(dtype_)->ToString();
  }
  data_size_ = type_pair->second.second;
  slice_param_.data_type = type_pair->second.first;

  for (size_t i = 0; i < DIMENSION_8D; i++) {
    int dim_len;
    if (i < begin.size()) {
      dim_len = SizeToInt(input_shape_[i]);
      int begin_pos = LongToInt(begin[i]);
      int end_pos = LongToInt(end[i]);
      int stride_size = LongToInt(stride[i]);
      if (stride_size == 0) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the each dimension slice stride should not be 0.";
      }
      slice_param_.in_shape_[i] = dim_len;
      slice_param_.strides_[i] = stride_size;
      slice_param_.begins_[i] = NormalizePos(begin_pos, dim_len, kBegin);
      slice_param_.ends_[i] = NormalizePos(end_pos, dim_len, kEnd);
      if (slice_param_.ends_[i] <= slice_param_.begins_[i] && slice_param_.strides_[i] > 0) {
        slice_param_.ends_[i] = slice_param_.begins_[i] + 1;
      }
      if (slice_param_.ends_[i] >= slice_param_.begins_[i] && slice_param_.strides_[i] < 0) {
        slice_param_.ends_[i] = slice_param_.begins_[i] - 1;
      }
    } else if (i < input_shape_.size()) {
      dim_len = SizeToInt(input_shape_[i]);
      slice_param_.in_shape_[i] = dim_len;
      slice_param_.begins_[i] = 0;
      slice_param_.ends_[i] = dim_len;
      slice_param_.strides_[i] = 1;
    } else {
      slice_param_.in_shape_[i] = 1;
      slice_param_.begins_[i] = 0;
      slice_param_.ends_[i] = 1;
      slice_param_.strides_[i] = 1;
    }
  }
  slice_param_.in_shape_length_ = DIMENSION_8D;
  slice_param_.num_axes_ = DIMENSION_8D;
}

int StridedSliceCPUKernel::RunTaskOnOuter(const uint8_t *input_addr, uint8_t *output_addr, int start_pos) {
  int begin_index = slice_param_.begins_[split_axis_];
  int inner_size = inner_ * data_size_;
  const uint8_t *cur_in_ptr = input_addr + (start_pos * input_shape_[split_axis_] + begin_index) * inner_size;
  uint8_t *cur_out_ptr = output_addr + start_pos * output_shape_[split_axis_] * inner_size;
  int cur_outer = outer_ - start_pos;
  if (cur_outer <= 0) {
    return common::SUCCESS;
  }
  cur_outer = cur_outer > cal_num_per_thread_ ? cal_num_per_thread_ : cur_outer;
  FastStride(cur_in_ptr, cur_out_ptr, output_shape_[split_axis_], slice_param_.strides_[split_axis_], cur_outer,
             inner_size, input_shape_[split_axis_] * inner_size);
  return common::SUCCESS;
}

int StridedSliceCPUKernel::RunTaskOnSplitAxis(const uint8_t *input_addr, uint8_t *output_addr, int start_pos) {
  int begin_index = slice_param_.begins_[split_axis_];
  int inner_size = inner_ * data_size_;
  const uint8_t *cur_in_ptr = input_addr + (start_pos * slice_param_.strides_[split_axis_] + begin_index) * inner_size;
  uint8_t *cur_out_ptr = output_addr + start_pos * inner_size;
  int cal_axis_num = output_shape_[split_axis_] - start_pos;
  if (cal_axis_num <= 0) {
    return common::SUCCESS;
  }
  cal_axis_num = cal_axis_num > cal_num_per_thread_ ? cal_num_per_thread_ : cal_axis_num;
  FastStride(cur_in_ptr, cur_out_ptr, cal_axis_num, slice_param_.strides_[split_axis_], 1, inner_size, 0);
  return common::SUCCESS;
}

void StridedSliceCPUKernel::ParallelRun(const uint8_t *input_addr, uint8_t *output_addr, int thread_num) {
  int thread_index = 0;
  std::vector<common::Task> tasks;
  std::function<int(StridedSliceCPUKernel *, const uint8_t *, uint8_t *, int)> execute_func;
  if (parallel_strategy_ == kOnOuter) {
    execute_func = &StridedSliceCPUKernel::RunTaskOnOuter;
  } else if (parallel_strategy_ == kOnSplitAxis) {
    execute_func = &StridedSliceCPUKernel::RunTaskOnSplitAxis;
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', not supports parallel execute strategy.";
  }

  while (thread_index < thread_num) {
    (void)tasks.emplace_back(
      std::bind(execute_func, this, input_addr, output_addr, thread_index * cal_num_per_thread_));
    thread_index++;
  }
  (void)common::ThreadPool::GetInstance().SyncRun(tasks);
}

bool StridedSliceCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                   const std::vector<kernel::AddressPtr> & /* workspace */,
                                   const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.size() != kStridedSliceInputsNum && inputs.size() != kStridedSliceDynamicInputsNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be " << kStridedSliceInputsNum
                      << " or " << kStridedSliceDynamicInputsNum << ", but got " << inputs.size();
  }
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kStridedSliceOutputsNum, kernel_name_);
  auto input_addr = reinterpret_cast<uint8_t *>(inputs[0]->addr);
  auto output_addr = reinterpret_cast<uint8_t *>(outputs[0]->addr);

  auto cnode = cnode_ptr_.lock();
  size_t input_num = AnfAlgo::GetInputTensorNum(cnode);
  if (input_num == kStridedSliceDynamicInputsNum) {
    // for begin, end, stride are not const input
    auto begin_shape = AnfAlgo::GetPrevNodeOutputInferShape(cnode, 1);
    auto end_shape = AnfAlgo::GetPrevNodeOutputInferShape(cnode, 2);
    auto stride_shape = AnfAlgo::GetPrevNodeOutputInferShape(cnode, 3);
    if (begin_shape.size() != 1 || end_shape.size() != 1 || stride_shape.size() != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the dimension of 'begin', 'end', 'strides' should be equal "
                           "to 1, but got the dimension of 'begin': "
                        << begin_shape.size() << ", the dimension of 'end': " << end_shape.size()
                        << ", and the dimension of 'strides': " << stride_shape.size();
    }
    auto begin_ptr = reinterpret_cast<int64_t *>(inputs[1]->addr);
    auto end_ptr = reinterpret_cast<int64_t *>(inputs[2]->addr);
    auto strides_ptr = reinterpret_cast<int64_t *>(inputs[3]->addr);
    std::vector<int64_t> begin{begin_ptr, begin_ptr + begin_shape[0]};
    std::vector<int64_t> end{end_ptr, end_ptr + end_shape[0]};
    std::vector<int64_t> stride{strides_ptr, strides_ptr + stride_shape[0]};
    if (begin.size() != end.size() || begin.size() != stride.size() || begin.size() > input_shape_.size()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the length of 'begin', 'stride' and 'end' should be equal "
                           "and less than or equal to the dimension of 'input_x', but got the length of 'begin': "
                        << begin.size() << ", the length of 'stride': " << stride.size()
                        << ", the length of 'end': " << end.size()
                        << ", the dimension of 'input_x': " << input_shape_.size();
    }
    InitSliceParam(begin, end, stride);
    parallel_ = MatchParallelPattern();
    if (parallel_) {
      InitParallelParam();
    }
  }

  int thread_num = slice_param_.op_parameter_.thread_num_;
  if (parallel_ && thread_num >= 2) {
    ParallelRun(input_addr, output_addr, thread_num);
  } else {
    (void)DoStridedSlice(input_addr, output_addr, &slice_param_);
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
