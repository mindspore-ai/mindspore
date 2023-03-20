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

#include "plugin/device/cpu/kernel/tensor_copy_slices_cpu_kernel.h"
#include <algorithm>
#include <functional>
#include <unordered_map>
#include "abstract/utils.h"
#include "kernel/common_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kTensorCopySlicesInputsNum = 2;
constexpr size_t kTensorCopySlicesDynamicInputsNum = 5;
constexpr size_t kTensorCopySlicesOutputsNum = 1;
constexpr auto kBeginIdx = 2;
constexpr auto kEndIdx = 3;
constexpr auto kStridesIdx = 4;
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

bool TensorCopySlicesCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  data_type_ = inputs.at(kIndex0)->GetDtype();
  return true;
}

int TensorCopySlicesCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs,
                                         const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  input_shape_ = inputs.at(kIndex0)->GetShapeVector();
  update_shape_ = inputs.at(kIndex1)->GetShapeVector();
  output_shape_ = outputs.at(kIndex0)->GetShapeVector();
  begin_shape_ = inputs.at(kIndex2)->GetShapeVector();
  end_shape_ = inputs.at(kIndex3)->GetShapeVector();
  stride_shape_ = inputs.at(kIndex4)->GetShapeVector();
  get_value_before_launch_ = false;
  std::vector<int64_t> begin, end, stride;
  auto get_begin = TryGetIntValue(inputs, kBeginIdx, kernel_name_, &begin, false);
  auto get_end = TryGetIntValue(inputs, kEndIdx, kernel_name_, &end, false);
  auto get_stride = TryGetIntValue(inputs, kStridesIdx, kernel_name_, &stride, false);
  if (get_begin && get_end && get_stride) {
    FillSlice(&begin, &end);
    InitOffsetAndCopySize(begin, end, stride);
    get_value_before_launch_ = true;
  }
  return KRET_OK;
}

bool TensorCopySlicesCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                          const std::vector<kernel::AddressPtr> & /* workspace */,
                                          const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kTensorCopySlicesDynamicInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kTensorCopySlicesOutputsNum, kernel_name_);

  auto input_addr = reinterpret_cast<uint8_t *>(inputs[0]->addr);
  auto update_addr = reinterpret_cast<uint8_t *>(inputs[1]->addr);
  auto output_addr = reinterpret_cast<uint8_t *>(outputs[0]->addr);
  if (!get_value_before_launch_) {
    auto begin_ptr = GetDeviceAddress<int64_t>(inputs, kIndex2);
    auto end_ptr = GetDeviceAddress<int64_t>(inputs, kIndex3);
    auto strides_ptr = GetDeviceAddress<int64_t>(inputs, kIndex4);
    std::vector<int64_t> begin{begin_ptr, begin_ptr + begin_shape_[0]};
    std::vector<int64_t> end{end_ptr, end_ptr + end_shape_[0]};
    std::vector<int64_t> stride{strides_ptr, strides_ptr + stride_shape_[0]};
    FillSlice(&begin, &end);
    InitOffsetAndCopySize(begin, end, stride);
  }

  auto ret = memcpy_s(output_addr, outputs[0]->size, input_addr, inputs[0]->size);
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
}  // namespace kernel
}  // namespace mindspore
