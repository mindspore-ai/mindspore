/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/stridedslice_cpu_kernel.h"
#include <utility>
#include <functional>
#include <algorithm>
#include <unordered_map>
#include "include/common/thread_pool.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "kernel/common_utils.h"
#include "ops/strided_slice.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kStridedSliceInputsNum = 1;
constexpr size_t kStridedSliceDynamicInputsNum = 4;
constexpr size_t kStridedSliceOutputsNum = 1;
}  // namespace

bool StridedSliceCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  base_operator_ = base_operator;
  return true;
}

int StridedSliceCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs,
                                     const std::map<uint32_t, tensor::TensorPtr> &) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  if (inputs.size() != kStridedSliceInputsNum && inputs.size() != kStridedSliceDynamicInputsNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be " << kStridedSliceInputsNum
                      << " or " << kStridedSliceDynamicInputsNum << ", but got " << inputs.size();
  }
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kStridedSliceOutputsNum, kernel_name_);
  input_shape_ = inputs[0]->GetShapeVector();
  dtype_ = inputs[0]->GetDtype();
  output_shape_ = outputs[0]->GetShapeVector();
  if (input_shape_.size() > DIMENSION_8D || input_shape_.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'input_x' must be in range [1D, 8D], but got "
                  << input_shape_.size() << "D.";
    return KRET_RESIZE_FAILED;
  }
  parallel_ = MatchParallelPattern();
  if (parallel_) {
    InitParallelParam();
  }

  if (inputs.size() == kStridedSliceDynamicInputsNum) {
    // for begin, end, stride are not const input
    begin_shape_ = inputs[kIndex1]->GetShapeVector();
    end_shape_ = inputs[kIndex2]->GetShapeVector();
    stride_shape_ = inputs[kIndex3]->GetShapeVector();
    if (begin_shape_.size() != 1 || end_shape_.size() != 1 || stride_shape_.size() != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the dimension of 'begin', 'end', 'strides' must be equal "
                           "to 1, but got the dimension of 'begin': "
                        << begin_shape_.size() << ", the dimension of 'end': " << end_shape_.size()
                        << ", and the dimension of 'strides': " << stride_shape_.size();
    }
  } else {
    // for begin, end, stride are tuples
    auto kernel_ptr = std::dynamic_pointer_cast<ops::StridedSlice>(base_operator);
    auto begin = kernel_ptr->get_begin();
    auto end = kernel_ptr->get_end();
    auto stride = kernel_ptr->get_strides();
    InitSliceParam(base_operator, &begin, &end, &stride);
  }
  return ret;
}

bool StridedSliceCpuKernelMod::MatchParallelPattern() {
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

void StridedSliceCpuKernelMod::InitParallelParam() {
  auto idx = IntToSize(split_axis_);
  outer_ = LongToInt(std::accumulate(input_shape_.begin(), input_shape_.begin() + IntToLong(split_axis_), int64_t(1),
                                     std::multiplies<int64_t>()));
  inner_ = LongToInt(std::accumulate(input_shape_.begin() + IntToLong(split_axis_) + 1, input_shape_.end(), int64_t(1),
                                     std::multiplies<int64_t>()));

  auto thread_pool = GetActorMgrInnerThreadPool();
  int max_thread_num = SizeToInt(thread_pool->GetKernelThreadNum());
  int thread_num = 1;
  if (outer_ == 1) {
    parallel_strategy_ = kOnSplitAxis;
    thread_num = std::max(thread_num, std::min(LongToInt(output_shape_[idx]), max_thread_num));
    cal_num_per_thread_ = UP_DIV(output_shape_[idx], thread_num);
  } else {
    parallel_strategy_ = kOnOuter;
    thread_num = std::min(outer_, max_thread_num);
    cal_num_per_thread_ = UP_DIV(outer_, thread_num);
  }
  slice_param_.op_parameter_.thread_num_ = thread_num;
}

void StridedSliceCpuKernelMod::InitSliceParam(const BaseOperatorPtr &base_operator, std::vector<int64_t> *begin,
                                              std::vector<int64_t> *end, std::vector<int64_t> *stride) {
  static const std::unordered_map<TypeId, std::pair<TypeIdC, int>> type_convert_map = {
    {kNumberTypeBool, {::kNumberTypeBool, sizeof(bool)}},
    {kNumberTypeInt32, {::kNumberTypeInt32, sizeof(int)}},
    {kNumberTypeFloat32, {::kNumberTypeFloat32, sizeof(float)}},
    {kNumberTypeFloat64, {::kNumberTypeFloat64, sizeof(double)}}};

  auto type_pair = type_convert_map.find(dtype_);
  if (type_pair == type_convert_map.end()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dtype of 'input_x' must be bool, int32, float32 or float64, but got "
                      << TypeIdToType(dtype_)->ToString();
  }
  data_size_ = type_pair->second.second;
  slice_param_.data_type = type_pair->second.first;
  auto input_shape_pad = input_shape_;
  FillEmptyDims(base_operator, begin, end, stride, &input_shape_pad);
  ParseStrideSliceMasks(base_operator, begin, end, stride, input_shape_pad);

  std::vector<int64_t> &_begin = *begin;
  std::vector<int64_t> &_end = *end;
  std::vector<int64_t> &_stride = *stride;
  for (size_t i = 0; i < DIMENSION_8D; i++) {
    slice_param_.in_shape_[i] = LongToInt(input_shape_pad[i]);
    slice_param_.begins_[i] = LongToInt(_begin[i]);
    slice_param_.ends_[i] = LongToInt(_end[i]);
    slice_param_.strides_[i] = LongToInt(_stride[i]);
  }
  slice_param_.in_shape_length_ = DIMENSION_8D;
  slice_param_.num_axes_ = DIMENSION_8D;
}

common::Status StridedSliceCpuKernelMod::RunTaskOnOuter(const uint8_t *input_addr, uint8_t *output_addr,
                                                        int start_pos) {
  auto idx = IntToSize(split_axis_);
  int begin_index = slice_param_.begins_[idx];
  int inner_size = inner_ * data_size_;
  const uint8_t *cur_in_ptr =
    input_addr + IntToSize((start_pos * LongToInt(input_shape_[idx]) + begin_index) * inner_size);
  uint8_t *cur_out_ptr = output_addr + IntToSize(start_pos * LongToInt(output_shape_[idx]) * inner_size);
  int cur_outer = outer_ - start_pos;
  if (cur_outer <= 0) {
    return common::SUCCESS;
  }
  cur_outer = cur_outer > cal_num_per_thread_ ? cal_num_per_thread_ : cur_outer;
  FastStride(cur_in_ptr, cur_out_ptr, LongToInt(output_shape_[idx]), slice_param_.strides_[idx], cur_outer, inner_size,
             LongToSize(input_shape_[idx]) * inner_size);
  return common::SUCCESS;
}

common::Status StridedSliceCpuKernelMod::RunTaskOnSplitAxis(const uint8_t *input_addr, uint8_t *output_addr,
                                                            int start_pos) {
  auto idx = IntToSize(split_axis_);
  int begin_index = slice_param_.begins_[idx];
  int inner_size = inner_ * data_size_;
  const uint8_t *cur_in_ptr = input_addr + (start_pos * slice_param_.strides_[idx] + begin_index) * inner_size;
  uint8_t *cur_out_ptr = output_addr + start_pos * inner_size;
  int cal_axis_num = LongToInt(output_shape_[idx]) - start_pos;
  if (cal_axis_num <= 0) {
    return common::SUCCESS;
  }
  cal_axis_num = cal_axis_num > cal_num_per_thread_ ? cal_num_per_thread_ : cal_axis_num;
  FastStride(cur_in_ptr, cur_out_ptr, cal_axis_num, slice_param_.strides_[idx], 1, inner_size, 0);
  return common::SUCCESS;
}

void StridedSliceCpuKernelMod::ParallelRun(const uint8_t *input_addr, uint8_t *output_addr, int thread_num) {
  int thread_index = 0;
  std::vector<common::Task> tasks;
  std::function<common::Status(StridedSliceCpuKernelMod *, const uint8_t *, uint8_t *, int)> execute_func;
  if (parallel_strategy_ == kOnOuter) {
    execute_func = &StridedSliceCpuKernelMod::RunTaskOnOuter;
  } else if (parallel_strategy_ == kOnSplitAxis) {
    execute_func = &StridedSliceCpuKernelMod::RunTaskOnSplitAxis;
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', not supports parallel execute strategy.";
  }

  while (thread_index < thread_num) {
    (void)tasks.emplace_back(
      std::bind(execute_func, this, input_addr, output_addr, thread_index * cal_num_per_thread_));
    thread_index++;
  }
  ParallelLaunch(tasks);
}

bool StridedSliceCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                      const std::vector<kernel::AddressPtr> & /* workspace */,
                                      const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.size() != kStridedSliceInputsNum && inputs.size() != kStridedSliceDynamicInputsNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be " << kStridedSliceInputsNum
                      << " or " << kStridedSliceDynamicInputsNum << ", but got " << inputs.size();
  }
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kStridedSliceOutputsNum, kernel_name_);
  auto input_addr = reinterpret_cast<uint8_t *>(inputs[0]->addr);
  auto output_addr = reinterpret_cast<uint8_t *>(outputs[0]->addr);

  size_t input_num = inputs.size();
  if (input_num == kStridedSliceDynamicInputsNum) {
    // for begin, end, stride are tensors
    auto begin_ptr = reinterpret_cast<int64_t *>(inputs[kIndex1]->addr);
    auto end_ptr = reinterpret_cast<int64_t *>(inputs[kIndex2]->addr);
    auto strides_ptr = reinterpret_cast<int64_t *>(inputs[kIndex3]->addr);
    std::vector<int64_t> begin{begin_ptr, begin_ptr + begin_shape_[0]};
    std::vector<int64_t> end{end_ptr, end_ptr + end_shape_[0]};
    std::vector<int64_t> stride{strides_ptr, strides_ptr + stride_shape_[0]};
    InitSliceParam(base_operator_, &begin, &end, &stride);
  }

  int thread_num = slice_param_.op_parameter_.thread_num_;
  if (parallel_ && thread_num >= 2) {
    ParallelRun(input_addr, output_addr, thread_num);
  } else {
    (void)DoStridedSlice(input_addr, output_addr, &slice_param_);
  }
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, StridedSlice, StridedSliceCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
