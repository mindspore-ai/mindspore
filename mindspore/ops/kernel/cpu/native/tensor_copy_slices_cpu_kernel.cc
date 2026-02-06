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

#include "kernel/cpu/native/tensor_copy_slices_cpu_kernel.h"
#include <algorithm>
#include <functional>
#include <unordered_map>
#include "include/runtime/hardware_abstract/kernel_base/kernel_utils.h"

namespace mindspore {
namespace kernel {
namespace tensor_copy_slices_cpu {
namespace {
constexpr size_t kTensorCopySlicesInputsNum = 2;
constexpr size_t kTensorCopySlicesDynamicInputsNum = 5;
constexpr size_t kTensorCopySlicesOutputsNum = 1;
constexpr auto kBeginIdx = 2;
constexpr auto kEndIdx = 3;
constexpr auto kStridesIdx = 4;

std::vector<int64_t> CalDimOffset(const std::vector<int64_t> &input_shape) {
  std::vector<int64_t> dim_offset;
  int64_t offset = 1;
  for (auto iter = input_shape.rbegin(); iter != input_shape.rend(); ++iter) {
    dim_offset.push_back(offset);
    offset = offset * (*iter);
  }
  std::reverse(dim_offset.begin(), dim_offset.end());
  return dim_offset;
}

size_t CalOffset(const std::vector<int64_t> &start, const std::vector<int64_t> &stop,
                 const std::vector<int64_t> &dim_offset) {
  size_t size = start.size();
  size_t offset = 0;
  for (size_t i = 0; i < size; ++i) {
    offset += SizetMulWithOverflowCheck(LongToSize(dim_offset[i]), LongToSize(start[i]));
    if (stop[i] - start[i] != 1) {
      break;
    }
  }
  return offset;
}

void CheckSliceValid(const std::vector<int64_t> &start, const std::vector<int64_t> &stop,
                     const std::vector<int64_t> &step, const std::vector<int64_t> &input_shape) {
  if (start.size() != stop.size() || start.size() != step.size() || start.size() > input_shape.size()) {
    MS_LOG(EXCEPTION)
      << "TensorCopySlices requires the length of begin, stride and end must be equal and less than input dimension.";
  }

  size_t size = start.size();
  for (size_t i = 0; i < size; ++i) {
    if (stop[i] <= start[i]) {
      MS_LOG(EXCEPTION) << "Invalid slice: (" << start[i] << ", " << stop[i] << " ," << step[i] << ")";
    }
    // Operator need to be generalized in the future. Only support to copy continuous memory now.
    if (step[i] != 1) {
      MS_LOG(EXCEPTION) << "The element in step only support 1, but got:" << step;
    }
  }

  size_t slice_pos = size;
  for (size_t i = 0; i < size; ++i) {
    if (stop[i] - start[i] > 1) {
      slice_pos = i;
      break;
    }
  }

  for (size_t i = slice_pos + 1; i < size; ++i) {
    if (stop[i] - start[i] != input_shape[i]) {
      MS_LOG(EXCEPTION) << "Only support copy continuous memory now. For example tensor[0, 0:100] is fine, "
                           "but tensor[0:100, 0] is not supported.";
    }
  }
}

size_t GetCopySize(const std::vector<int64_t> &dim_offset, const std::vector<int64_t> &start,
                   const std::vector<int64_t> &stop) {
  for (size_t i = 0; i < start.size(); ++i) {
    if (stop[i] - start[i] != 1) {
      return SizetMulWithOverflowCheck(LongToSize(stop[i] - start[i]), LongToSize(dim_offset[i]));
    }
  }
  return LongToSize(dim_offset[start.size() - 1]);
}
}  // namespace

void TensorCopySlicesCpuKernelMod::FillSlice(std::vector<int64_t> *begin, std::vector<int64_t> *end) {
  std::vector<int64_t> &_begin = *begin;
  std::vector<int64_t> &_end = *end;
  if (_begin.size() != _end.size()) {
    MS_LOG(EXCEPTION)
      << "For '" << kernel_name_ << ","
      << "TensorCopySlices requires the length of begin, end must be equal and less than input dimension.";
  }
  for (size_t i = 0; i < _begin.size(); i++) {
    int64_t dim = input_shape_[i];
    _begin[i] = std::min(_begin[i] < 0 ? std::max(_begin[i] + dim, static_cast<int64_t>(0)) : _begin[i], dim - 1);
    _end[i] = std::max(_end[i] < 0 ? _end[i] + dim : std::min(_end[i], dim), static_cast<int64_t>(-1));
  }
}

void TensorCopySlicesCpuKernelMod::InitOffsetAndCopySize(const std::vector<int64_t> &begin,
                                                         const std::vector<int64_t> &end,
                                                         const std::vector<int64_t> &stride) {
  CheckSliceValid(begin, end, stride, input_shape_);

  auto dim_offset = CalDimOffset(input_shape_);
  auto type_size = abstract::TypeIdSize(data_type_);
  offset_ = CalOffset(begin, end, dim_offset) * type_size;
  copy_size_ = GetCopySize(dim_offset, begin, end) * type_size;
}

bool TensorCopySlicesCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  data_type_ = inputs.at(kIndex0)->dtype_id();
  return true;
}

int TensorCopySlicesCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  input_shape_ = inputs.at(kIndex0)->GetShapeVector();
  update_shape_ = inputs.at(kIndex1)->GetShapeVector();
  output_shape_ = outputs.at(kIndex0)->GetShapeVector();
  begin_shape_ = inputs.at(kIndex2)->GetShapeVector();
  end_shape_ = inputs.at(kIndex3)->GetShapeVector();
  stride_shape_ = inputs.at(kIndex4)->GetShapeVector();
  get_value_before_launch_ = false;
  auto begin = inputs[kBeginIdx]->GetValueWithCheck<std::vector<int64_t>>();
  auto end = inputs[kEndIdx]->GetValueWithCheck<std::vector<int64_t>>();
  auto stride = inputs[kStridesIdx]->GetValueWithCheck<std::vector<int64_t>>();
  if (!begin.empty() && !end.empty() && !stride.empty()) {
    FillSlice(&begin, &end);
    InitOffsetAndCopySize(begin, end, stride);
    get_value_before_launch_ = true;
  }
  return KRET_OK;
}

bool TensorCopySlicesCpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                                          const std::vector<kernel::KernelTensor *> & /* workspace */,
                                          const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kTensorCopySlicesDynamicInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kTensorCopySlicesOutputsNum, kernel_name_);

  auto input_addr = reinterpret_cast<uint8_t *>(inputs[0]->device_ptr());
  auto update_addr = reinterpret_cast<uint8_t *>(inputs[1]->device_ptr());
  auto output_addr = reinterpret_cast<uint8_t *>(outputs[0]->device_ptr());
  if (!get_value_before_launch_) {
    auto begin_ptr = GetDeviceAddress<int64_t>(inputs, kIndex2);
    MS_EXCEPTION_IF_NULL(begin_ptr);
    auto end_ptr = GetDeviceAddress<int64_t>(inputs, kIndex3);
    MS_EXCEPTION_IF_NULL(end_ptr);
    auto strides_ptr = GetDeviceAddress<int64_t>(inputs, kIndex4);
    MS_EXCEPTION_IF_NULL(strides_ptr);
    std::vector<int64_t> begin{begin_ptr, begin_ptr + begin_shape_[0]};
    std::vector<int64_t> end{end_ptr, end_ptr + end_shape_[0]};
    std::vector<int64_t> stride{strides_ptr, strides_ptr + stride_shape_[0]};
    FillSlice(&begin, &end);
    InitOffsetAndCopySize(begin, end, stride);
  }

  auto ret = memcpy_s(output_addr, outputs[0]->size(), input_addr, inputs[0]->size());
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memcpy input failed. Error no: " << ret;
  }
  ret = memcpy_s(output_addr + offset_, copy_size_, update_addr, copy_size_);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memcpy update failed. Error no: " << ret;
  }
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, TensorCopySlices, TensorCopySlicesCpuKernelMod);
}  // namespace tensor_copy_slices_cpu
}  // namespace kernel
}  // namespace mindspore
