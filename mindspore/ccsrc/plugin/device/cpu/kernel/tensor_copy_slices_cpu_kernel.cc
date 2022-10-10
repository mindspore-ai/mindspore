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

void TensorCopySlicesCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  input_shape_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  update_shape_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  output_shape_ = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);
  data_type_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  cnode_ptr_ = kernel_node;

  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != kTensorCopySlicesInputsNum) {
    return;
  }

  auto begin = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, BEGIN);
  auto end = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, END);
  auto stride = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, STRIDES);
  FillSlice(&begin, &end);
  CheckSliceValid(begin, end, stride, input_shape_);

  auto dim_offset = CalDimOffset(input_shape_);
  auto type_size = abstract::TypeIdSize(data_type_);
  offset_ = CalOffset(begin, end, dim_offset) * type_size;
  copy_size_ = GetCopySize(dim_offset, begin, end) * type_size;
}

bool TensorCopySlicesCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                          const std::vector<kernel::AddressPtr> & /* workspace */,
                                          const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kTensorCopySlicesOutputsNum, kernel_name_);

  auto input_addr = reinterpret_cast<uint8_t *>(inputs[0]->addr);
  auto update_addr = reinterpret_cast<uint8_t *>(inputs[1]->addr);
  auto output_addr = reinterpret_cast<uint8_t *>(outputs[0]->addr);
  auto cnode = cnode_ptr_.lock();
  size_t input_num = common::AnfAlgo::GetInputTensorNum(cnode);
  if (input_num == kTensorCopySlicesDynamicInputsNum) {
    auto begin_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(cnode, 2);
    auto end_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(cnode, 3);
    auto stride_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(cnode, 4);
    if (begin_shape.size() != 1 || end_shape.size() != 1 || stride_shape.size() != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the dimension of 'begin', 'end', 'strides' must be equal "
                           "to 1, but got the dimension of 'begin': "
                        << begin_shape.size() << ", the dimension of 'end': " << end_shape.size()
                        << ", and the dimension of 'strides': " << stride_shape.size();
    }
    auto begin_ptr = static_cast<int64_t *>(inputs[2]->addr);
    auto end_ptr = static_cast<int64_t *>(inputs[3]->addr);
    auto strides_ptr = static_cast<int64_t *>(inputs[4]->addr);
    std::vector<int64_t> begin{begin_ptr, begin_ptr + begin_shape[0]};
    std::vector<int64_t> end{end_ptr, end_ptr + end_shape[0]};
    std::vector<int64_t> stride{strides_ptr, strides_ptr + stride_shape[0]};
    FillSlice(&begin, &end);
    CheckSliceValid(begin, end, stride, input_shape_);
    auto dim_offset = CalDimOffset(input_shape_);
    auto type_size = abstract::TypeIdSize(data_type_);
    offset_ = CalOffset(begin, end, dim_offset) * type_size;
    copy_size_ = GetCopySize(dim_offset, begin, end) * type_size;
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
