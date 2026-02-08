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

#include "kernel/cpu/native/stridedslice_cpu_kernel.h"
#include <utility>
#include <functional>
#include <algorithm>
#include <unordered_map>
#include <complex>
#include "include/utils/thread_pool.h"

#include "include/runtime/hardware_abstract/kernel_base/kernel_utils.h"

namespace mindspore {
namespace kernel {
namespace stridedslice_cpu {
namespace {
constexpr size_t kStridedSliceInputsNum = 9;
constexpr size_t kStridedSliceOutputsNum = 1;

using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
}  // namespace

bool StridedSliceCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int StridedSliceCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  auto ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  if (inputs.size() != kStridedSliceInputsNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be " << kStridedSliceInputsNum
                      << ", but got " << inputs.size();
  }

  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kStridedSliceOutputsNum, kernel_name_);
  input_shape_ = inputs[kIndex0]->GetShapeVector();
  dtype_ = inputs[kIndex0]->dtype_id();
  output_shape_ = outputs[kIndex0]->GetShapeVector();
  if (input_shape_.size() > DIMENSION_8D || input_shape_.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'input_x' must be in range [1D, 8D], but got "
                  << input_shape_.size() << "D.";
    return KRET_RESIZE_FAILED;
  }
  parallel_ = MatchParallelPattern();
  if (parallel_) {
    InitParallelParam();
  }

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
  slice_struct_.base_.thread_nr_ = thread_num;
}

void StridedSliceCpuKernelMod::InitSliceParam(std::vector<int64_t> *begin, std::vector<int64_t> *end,
                                              std::vector<int64_t> *stride,
                                              const std::vector<kernel::KernelTensor *> &inputs) {
  static const std::unordered_map<TypeId, std::pair<TypeIdC, int>> type_convert_map = {
    {kNumberTypeBool, {::kNumberTypeBool, sizeof(bool)}},
    {kNumberTypeInt8, {::kNumberTypeInt8, sizeof(int8_t)}},
    {kNumberTypeInt16, {::kNumberTypeInt16, sizeof(int16_t)}},
    {kNumberTypeInt32, {::kNumberTypeInt32, sizeof(int32_t)}},
    {kNumberTypeInt64, {::kNumberTypeInt64, sizeof(int64_t)}},
    {kNumberTypeUInt8, {::kNumberTypeUInt8, sizeof(uint8_t)}},
    {kNumberTypeUInt16, {::kNumberTypeUInt16, sizeof(uint16_t)}},
    {kNumberTypeUInt32, {::kNumberTypeUInt32, sizeof(uint32_t)}},
    {kNumberTypeUInt64, {::kNumberTypeUInt64, sizeof(uint64_t)}},
    {kNumberTypeFloat32, {::kNumberTypeFloat32, sizeof(float)}},
    {kNumberTypeFloat64, {::kNumberTypeFloat64, sizeof(double)}},
    {kNumberTypeComplex64, {::kNumberTypeComplex64, sizeof(complex64)}},
    {kNumberTypeComplex128, {::kNumberTypeComplex128, sizeof(complex128)}}};

  auto type_pair = type_convert_map.find(dtype_);
  if (type_pair == type_convert_map.end()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of 'input_x' is " << TypeIdToType(dtype_)->ToString()
                      << ", which is not supported.";
  }
  data_size_ = type_pair->second.second;
  slice_struct_.data_type_ = type_pair->second.first;
  auto input_shape_pad = input_shape_;
  FillEmptyDims(kernel_name_, begin, end, stride, &input_shape_pad);
  ParseStrideSliceMasks(inputs, begin, end, stride, input_shape_pad);

  std::vector<int64_t> &_begin = *begin;
  std::vector<int64_t> &_end = *end;
  std::vector<int64_t> &_stride = *stride;
  for (size_t i = 0; i < DIMENSION_8D; i++) {
    slice_struct_.in_shape_[i] = LongToInt(input_shape_pad[i]);
    slice_struct_.begins_[i] = LongToInt(_begin[i]);
    slice_struct_.ends_[i] = LongToInt(_end[i]);
    slice_struct_.strides_[i] = LongToInt(_stride[i]);
  }
}

common::Status StridedSliceCpuKernelMod::RunTaskOnOuter(const uint8_t *input_addr, uint8_t *output_addr,
                                                        int start_pos) {
  auto idx = IntToSize(split_axis_);
  int begin_index = slice_struct_.begins_[idx];
  int inner_size = inner_ * data_size_;
  const uint8_t *cur_in_ptr =
    input_addr + IntToSize((start_pos * LongToInt(input_shape_[idx]) + begin_index) * inner_size);
  uint8_t *cur_out_ptr = output_addr + IntToSize(start_pos * LongToInt(output_shape_[idx]) * inner_size);
  int cur_outer = outer_ - start_pos;
  if (cur_outer <= 0) {
    return common::SUCCESS;
  }
  cur_outer = cur_outer > cal_num_per_thread_ ? cal_num_per_thread_ : cur_outer;
  FastStride(cur_in_ptr, cur_out_ptr, LongToInt(output_shape_[idx]), slice_struct_.strides_[idx], cur_outer, inner_size,
             LongToSize(input_shape_[idx]) * inner_size);
  return common::SUCCESS;
}

common::Status StridedSliceCpuKernelMod::RunTaskOnSplitAxis(const uint8_t *input_addr, uint8_t *output_addr,
                                                            int start_pos) {
  auto idx = IntToSize(split_axis_);
  int begin_index = slice_struct_.begins_[idx];
  int inner_size = inner_ * data_size_;
  const uint8_t *cur_in_ptr = input_addr + (start_pos * slice_struct_.strides_[idx] + begin_index) * inner_size;
  uint8_t *cur_out_ptr = output_addr + start_pos * inner_size;
  int cal_axis_num = LongToInt(output_shape_[idx]) - start_pos;
  if (cal_axis_num <= 0) {
    return common::SUCCESS;
  }
  cal_axis_num = cal_axis_num > cal_num_per_thread_ ? cal_num_per_thread_ : cal_axis_num;
  FastStride(cur_in_ptr, cur_out_ptr, cal_axis_num, slice_struct_.strides_[idx], 1, inner_size, 0);
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

template <typename T, typename S>
bool StridedSliceCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                            const std::vector<kernel::KernelTensor *> & /* workspace */,
                                            const std::vector<kernel::KernelTensor *> &outputs) {
  if (inputs.size() != kStridedSliceInputsNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be " << kStridedSliceInputsNum
                      << ", but got " << inputs.size();
  }
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kStridedSliceOutputsNum, kernel_name_);
  auto input_addr = GetDeviceAddress<uint8_t>(inputs, kIndex0);
  auto output_addr = GetDeviceAddress<uint8_t>(outputs, kIndex0);

  // for begin, end, stride are tensors
  std::vector<int64_t> begin;
  std::vector<int64_t> end;
  std::vector<int64_t> stride;
  auto begin_ptr = GetDeviceAddress<S>(inputs, kIndex1);
  auto end_ptr = GetDeviceAddress<S>(inputs, kIndex2);
  auto strides_ptr = GetDeviceAddress<S>(inputs, kIndex3);
  for (int64_t i = 0; i < begin_shape_[kIndex0]; i++) {
    begin.push_back(static_cast<int64_t>(begin_ptr[i]));
  }
  for (int64_t i = 0; i < end_shape_[kIndex0]; i++) {
    end.push_back(static_cast<int64_t>(end_ptr[i]));
  }
  for (int64_t i = 0; i < stride_shape_[kIndex0]; i++) {
    stride.push_back(static_cast<int64_t>(strides_ptr[i]));
  }
  InitSliceParam(&begin, &end, &stride, inputs);

  int thread_num = slice_struct_.base_.thread_nr_;
  if (parallel_ && thread_num >= 2) {
    ParallelRun(input_addr, output_addr, thread_num);
  } else {
    (void)DoStridedSliceIn8D(input_addr, output_addr, &slice_struct_);
  }
  return true;
}

std::vector<KernelAttr> StridedSliceCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, StridedSliceCpuKernelMod::StridedSliceFunc> &pair) { return pair.first; });
  return support_list;
}

#define STRIDEDSLICE_CPU_REG(TYPEID_1, TYPEID_2, TYPE_1, TYPE_2) \
  KernelAttr()                                                   \
    .AddInputAttr(TYPEID_1)                                      \
    .AddInputAttr(kObjectTypeTuple, TYPEID_2)                    \
    .AddInputAttr(kObjectTypeTuple, TYPEID_2)                    \
    .AddInputAttr(kObjectTypeTuple, TYPEID_2)                    \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)           \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)           \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)           \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)           \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)           \
    .AddOutputAttr(TYPEID_1),                                    \
    &StridedSliceCpuKernelMod::LaunchKernel<TYPE_1, TYPE_2>

std::vector<std::pair<KernelAttr, StridedSliceCpuKernelMod::StridedSliceFunc>> StridedSliceCpuKernelMod::func_list_ = {
  {STRIDEDSLICE_CPU_REG(kNumberTypeFloat64, kNumberTypeInt64, double, int64_t)},
  {STRIDEDSLICE_CPU_REG(kNumberTypeFloat32, kNumberTypeInt64, float, int64_t)},
  {STRIDEDSLICE_CPU_REG(kNumberTypeInt64, kNumberTypeInt64, int64_t, int64_t)},
  {STRIDEDSLICE_CPU_REG(kNumberTypeInt32, kNumberTypeInt64, int32_t, int64_t)},
  {STRIDEDSLICE_CPU_REG(kNumberTypeInt16, kNumberTypeInt64, int16_t, int64_t)},
  {STRIDEDSLICE_CPU_REG(kNumberTypeInt8, kNumberTypeInt64, int8_t, int64_t)},
  {STRIDEDSLICE_CPU_REG(kNumberTypeUInt64, kNumberTypeInt64, uint64_t, int64_t)},
  {STRIDEDSLICE_CPU_REG(kNumberTypeUInt32, kNumberTypeInt64, uint32_t, int64_t)},
  {STRIDEDSLICE_CPU_REG(kNumberTypeUInt16, kNumberTypeInt64, uint16_t, int64_t)},
  {STRIDEDSLICE_CPU_REG(kNumberTypeUInt8, kNumberTypeInt64, uint8_t, int64_t)},
  {STRIDEDSLICE_CPU_REG(kNumberTypeBool, kNumberTypeInt64, bool, int64_t)},
  {STRIDEDSLICE_CPU_REG(kNumberTypeComplex64, kNumberTypeInt64, complex64, int64_t)},
  {STRIDEDSLICE_CPU_REG(kNumberTypeComplex128, kNumberTypeInt64, complex128, int64_t)},
};

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, StridedSlice, StridedSliceCpuKernelMod);
}  // namespace stridedslice_cpu
}  // namespace kernel
}  // namespace mindspore
