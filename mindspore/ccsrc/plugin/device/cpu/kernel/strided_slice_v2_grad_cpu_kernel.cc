/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/strided_slice_v2_grad_cpu_kernel.h"
#include <algorithm>
#include <string>
#include <map>
#include <complex>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "ir/primitive.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
constexpr size_t kStridedSliceV2GradDynamicInputsNum = 5;
constexpr size_t kOutputsNum = 1;
constexpr size_t kStridedSliceV2GradMaxInputShapeSize = 8;

static std::map<std::string, std::vector<KernelAttr>> support_list_map = {
  {kStridedSliceV2Grad,
   {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
    KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
    KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
    KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
    KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
    KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
    KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
    KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
    KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
    KernelAttr()
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeFloat16)
      .AddOutputAttr(kNumberTypeFloat16),
    KernelAttr()
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeFloat32)
      .AddOutputAttr(kNumberTypeFloat32),
    KernelAttr()
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeFloat64)
      .AddOutputAttr(kNumberTypeFloat64),
    KernelAttr()
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt8)
      .AddOutputAttr(kNumberTypeInt8),
    KernelAttr()
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt16)
      .AddOutputAttr(kNumberTypeInt16),
    KernelAttr()
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeInt32),
    KernelAttr()
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt64),
    KernelAttr()
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeUInt8)
      .AddOutputAttr(kNumberTypeUInt8),
    KernelAttr()
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeUInt16)
      .AddOutputAttr(kNumberTypeUInt16),
    KernelAttr()
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeUInt32)
      .AddOutputAttr(kNumberTypeUInt32),
    KernelAttr()
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeUInt64)
      .AddOutputAttr(kNumberTypeUInt64),
    KernelAttr()
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeComplex64)
      .AddOutputAttr(kNumberTypeComplex64),
    KernelAttr()
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeComplex128)
      .AddOutputAttr(kNumberTypeComplex128),
    KernelAttr()
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeBool)
      .AddOutputAttr(kNumberTypeBool),
    KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeFloat16)
      .AddOutputAttr(kNumberTypeFloat16),
    KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeFloat32)
      .AddOutputAttr(kNumberTypeFloat32),
    KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeFloat64)
      .AddOutputAttr(kNumberTypeFloat64),
    KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt8)
      .AddOutputAttr(kNumberTypeInt8),
    KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt16)
      .AddOutputAttr(kNumberTypeInt16),
    KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeInt32),
    KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt64),
    KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeUInt8)
      .AddOutputAttr(kNumberTypeUInt8),
    KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeUInt16)
      .AddOutputAttr(kNumberTypeUInt16),
    KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeUInt32)
      .AddOutputAttr(kNumberTypeUInt32),
    KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeUInt64)
      .AddOutputAttr(kNumberTypeUInt64),
    KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeComplex64)
      .AddOutputAttr(kNumberTypeComplex64),
    KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeComplex128)
      .AddOutputAttr(kNumberTypeComplex128),
    KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeBool)
      .AddOutputAttr(kNumberTypeBool)}}};

std::vector<bool> GradDec2Bin(const int64_t &mask) {
  auto mask_str = std::bitset<kStridedSliceV2GradMaxInputShapeSize>(mask).to_string();
  int64_t dim_idx = 0;
  std::vector<bool> result(kStridedSliceV2GradMaxInputShapeSize, false);
  for (int64_t i = static_cast<int64_t>(mask_str.size()) - static_cast<int64_t>(1); i >= 0; i--) {
    if (mask_str[i] == '1') {
      result[dim_idx] = true;
    }
    dim_idx++;
  }
  return result;
}

template <typename T>
void ParseStrideSliceGradMasksST(const CNodePtr &kernel_node, std::vector<T> *begin, std::vector<T> *end,
                                 std::vector<T> *stride, ShapeVector *input_shape, const ShapeVector output_shape,
                                 int shape_dim_output, int slice_len) {
  std::vector<T> &_begin_attr = *begin;
  std::vector<T> &_end_attr = *end;
  std::vector<T> &_stride_attr = *stride;
  auto begin_mask_int = common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, kAttrBeginMask);
  auto begin_mask = GradDec2Bin(begin_mask_int);
  auto end_mask_int = common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, kAttrEndMask);
  auto end_mask = GradDec2Bin(end_mask_int);
  auto ellipsis_mask_int = common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, kAttrEllipsisMask);
  auto ellipsis_mask = GradDec2Bin(ellipsis_mask_int);
  auto new_axis_mask_int = common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, kAttrNewAxisMask);
  auto new_axis_mask = GradDec2Bin(new_axis_mask_int);
  auto shrink_axis_mask_int = common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, kAttrShrinkAxisMask);
  auto shrink_axis_mask = GradDec2Bin(shrink_axis_mask_int);
  int i = 0;
  int j = 0;
  int count = 0;
  std::vector<T> begin_new;
  std::vector<T> end_new;
  std::vector<T> stride_new;
  while (i < shape_dim_output || j < slice_len) {
    T begin_j;
    T end_j;
    T stride_j;
    if (j < slice_len) {
      begin_j = _begin_attr[j];
      end_j = _end_attr[j];
      stride_j = _stride_attr[j];
      if (begin_mask[j]) {
        begin_j = _stride_attr[j] < 0 ? static_cast<T>(output_shape[i]) - 1 : 0;
      }
      if (end_mask[j]) {
        end_j = _stride_attr[j] < 0 ? -1 : static_cast<T>(output_shape[i]);
      }
      if (ellipsis_mask[j]) {
        begin_j = 0;
        end_j = static_cast<T>(output_shape[i]);
        stride_j = 1;
      }
      if (new_axis_mask[j]) {
        input_shape->erase(input_shape->begin() + j - count);
        j++;
        count++;
        continue;
      }
      if (shrink_axis_mask[j]) {
        input_shape->insert(input_shape->begin() + j, 1);
        end_j = _begin_attr[j] + 1;
        stride_j = 1;
      }
      if (end_j > output_shape[i]) {
        end_j = output_shape[i];
      }
    } else {
      begin_j = 0;
      end_j = static_cast<T>(output_shape[i]);
      stride_j = 1;
    }
    begin_new.push_back(begin_j);
    end_new.push_back(end_j);
    stride_new.push_back(stride_j);
    i++;
    j++;
  }
  _begin_attr.assign(begin_new.begin(), begin_new.end());
  _end_attr.assign(end_new.begin(), end_new.end());
  _stride_attr.assign(stride_new.begin(), stride_new.end());
}

template <typename T>
void FillEmptyDimsSTGrad(std::vector<T> *begin, std::vector<T> *end, std::vector<T> *stride, ShapeVector *input_shape,
                         ShapeVector *output_shape) {
  std::vector<T> &_begin = *begin;
  std::vector<T> &_end = *end;
  std::vector<T> &_stride = *stride;
  auto &_input_shape = *input_shape;
  auto &_output_shape = *output_shape;

  for (size_t i = 0; i < DIMENSION_8D; i++) {
    if (i >= _input_shape.size()) {
      _input_shape.push_back(1);
    }

    if (i >= _output_shape.size()) {
      _output_shape.push_back(1);
    }

    if (i < _begin.size()) {
      T dim = static_cast<T>(_output_shape[i]);
      _begin[i] = std::min(_begin[i] < 0 ? std::max(_begin[i] + dim, static_cast<T>(0)) : _begin[i], dim - 1);
    } else {
      _begin.push_back(0);
    }

    if (i >= _end.size()) {
      _end.push_back(i < _output_shape.size() ? static_cast<T>(_output_shape[i]) : 1);
    }

    if (i >= _stride.size()) {
      _stride.push_back(1);
    }
  }
}
}  // namespace

void StridedSliceV2GradCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  cnode_ptr_ = kernel_node;
  ClearVectors();
  auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, kIndex4);
  if (input_shape.size() > kStridedSliceV2GradMaxInputShapeSize) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input tensor must be 8D or lower, but got "
                      << input_shape.size() << "D.";
  }
  output_shape_ = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, kIndex4);
  dtype_grad_attr = AnfAlgo::GetInputDeviceDataType(kernel_node, 1);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num == kStridedSliceV2GradDynamicInputsNum) {  // Dynamic Shape
    return;
  }
  // in the case that begin, end, size, stride are const value.
  std::vector<int64_t> begin_me = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, BEGIN);
  (void)std::transform(begin_me.begin(), begin_me.end(), std::back_inserter(begin_),
                       [](const int64_t &value) { return LongToInt(value); });
  auto prim = common::AnfAlgo::GetCNodePrimitive(kernel_node);
  MS_EXCEPTION_IF_NULL(prim);
  auto strides = prim->GetAttr(STRIDES);

  std::vector<int64_t> strides_me = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, STRIDES);
  std::vector<int64_t> end_me = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, END);
  (void)std::transform(strides_me.begin(), strides_me.end(), std::back_inserter(strides_),
                       [](const int64_t &value) { return LongToInt(value); });
  (void)std::transform(end_me.begin(), end_me.end(), std::back_inserter(end_),
                       [](const int64_t &value) { return LongToInt(value); });
  if (strides_.size() != end_.size() || strides_.size() != output_shape_.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of 'strides|end|output' must be equal, but got the dimension of "
                      << "'strides': " << strides_.size() << ", the dimension of 'end': " << end_.size()
                      << ", and the dimension of output: " << output_shape_.size();
  }
  FormatArgs(true);
  ExpandAllMemberDims(kStridedSliceV2GradMaxInputShapeSize);
}

void StridedSliceV2GradCpuKernelMod::ClearVectors() {
  begin_.clear();
  size_.clear();
  strides_.clear();
  end_.clear();
  input_element_num_.clear();
  output_element_num_.clear();
  input_shape_.clear();
  output_shape_.clear();
}

void StridedSliceV2GradCpuKernelMod::ExpandAllMemberDims(size_t expand_dims) {
  auto output_len = output_shape_.size();
  auto strides_len = strides_.size();
  if (output_len < expand_dims) {
    for (size_t i = 0; i < expand_dims - output_len; ++i) {
      (void)output_shape_.insert(output_shape_.begin(), 1);
    }
  }

  if (strides_len < expand_dims) {
    for (size_t i = 0; i < expand_dims - strides_len; ++i) {
      (void)begin_.insert(begin_.begin(), 0);
      (void)strides_.insert(strides_.begin(), 1);
      (void)end_.insert(end_.begin(), 1);
    }
  }

  for (size_t i = 0; i < expand_dims; ++i) {
    if (SignOfStride(i) == 1) {
      int ax = (end_[i] - begin_[i]) * SignOfStride(i);
      if (ax < 0) {
        ax = 0;
      }
      input_shape_.push_back(ax);
    }
  }
}

// init for dynamic shape
template <typename T>
void StridedSliceV2GradCpuKernelMod::InitParams(const std::vector<kernel::AddressPtr> &inputs) {
  auto cnode = cnode_ptr_.lock();
  ClearVectors();
  output_shape_ = common::AnfAlgo::GetOutputInferShape(cnode, 0);
  std::string kernel_name = common::AnfAlgo::GetCNodeName(cnode);
  auto begin_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(cnode, 1);
  auto begin_ptr = static_cast<T *>(inputs[1]->addr);
  std::vector<T> begin{begin_ptr, begin_ptr + begin_shape[0]};

  auto end_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(cnode, kIndex2);
  auto stride_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(cnode, kIndex3);
  if (begin_shape.size() != 1 || end_shape.size() != 1 || stride_shape.size() != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimensions of 'begin', 'end', 'strides' must be 1, "
                         "but got the dimension of 'begin': "
                      << begin_shape.size() << ", the dimension of 'end': " << end_shape.size()
                      << ", and the dimension of 'strides': " << stride_shape.size();
  }
  auto end_ptr = static_cast<T *>(inputs[kIndex2]->addr);
  auto strides_ptr = static_cast<T *>(inputs[kIndex3]->addr);

  std::vector<T> end{end_ptr, end_ptr + end_shape[0]};
  std::vector<T> strides{strides_ptr, strides_ptr + stride_shape[0]};
  input_shape_ = common::AnfAlgo::GetPrevNodeOutputInferShape(cnode, kIndex4);
  shape_dim_output = SizeToInt(output_shape_.size());
  slice_len = SizeToInt(begin.size());
  FillEmptyDimsSTGrad<T>(&begin, &end, &strides, &input_shape_, &output_shape_);
  ParseStrideSliceGradMasksST<T>(cnode, &begin, &end, &strides, &input_shape_, output_shape_, shape_dim_output,
                                 slice_len);
  FillEmptyDimsSTGrad<T>(&begin, &end, &strides, &input_shape_, &output_shape_);
  (void)std::transform(begin.begin(), begin.end(), std::back_inserter(begin_), [](const T &value) { return value; });
  (void)std::transform(strides.begin(), strides.end(), std::back_inserter(strides_),
                       [](const T &value) { return value; });
  (void)std::transform(end.begin(), end.end(), std::back_inserter(end_), [](const T &value) { return value; });
  if (strides_.size() != end_.size() || strides_.size() != output_shape_.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of 'strides|end|output' must be equal, but got the dimension of "
                      << "'strides': " << strides_.size() << ", the dimension of 'end': " << end_.size()
                      << ", and the dimension of output: " << output_shape_.size();
  }
  FormatArgs(true);
}

bool StridedSliceV2GradCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                            const std::vector<kernel::AddressPtr> &,
                                            const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.empty()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', input can not be empty.";
  }
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  bool ret = true;
  if (dtype_ == kNumberTypeInt32) {
    ret = LaunchKernel<int32_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt8) {
    ret = LaunchKernel<int8_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt16) {
    ret = LaunchKernel<int16_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt64) {
    ret = LaunchKernel<int64_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeUInt8) {
    ret = LaunchKernel<uint8_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeUInt16) {
    ret = LaunchKernel<uint16_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeUInt64) {
    ret = LaunchKernel<uint64_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeUInt32) {
    ret = LaunchKernel<uint32_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat16) {
    ret = LaunchKernel<float16>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    ret = LaunchKernel<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat64) {
    ret = LaunchKernel<double>(inputs, outputs);
  } else if (dtype_ == kNumberTypeBool) {
    ret = LaunchKernel<bool>(inputs, outputs);
  } else if (dtype_ == kNumberTypeComplex64) {
    ret = LaunchKernel<complex64>(inputs, outputs);
  } else if (dtype_ == kNumberTypeComplex128) {
    ret = LaunchKernel<complex128>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dtype of input must be bool, int8, int16, int32, int64, uint8, uint16, uint32, "
                         "uint64, float16, float32, float64, complex64 or complex128, but got "
                      << TypeIdToType(dtype_)->ToString();
  }
  return ret;
}

template <typename T>
bool StridedSliceV2GradCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                  const std::vector<kernel::AddressPtr> &outputs) {
  auto *input_addr = static_cast<T *>(inputs[kIndex4]->addr);
  auto *output_addr = static_cast<T *>(outputs[0]->addr);
  if (inputs.size() == kStridedSliceV2GradDynamicInputsNum) {
    if (dtype_grad_attr == kNumberTypeInt32) {
      InitParams<int32_t>(inputs);
    } else {
      InitParams<int64_t>(inputs);
    }
  }
  return CalStridedSliceV2Grad<T>(input_addr, output_addr);
}

template <typename T>
bool StridedSliceV2GradCpuKernelMod::CalStridedSliceV2Grad(T *input, T *output) {
  int length = SizeToInt(input_shape_.size());
  int input_num = 1;
  for (int le = 0; le < length; le++) {
    input_num = input_num * input_shape_[le];
  }
  int output_num = 1;
  for (int len = 0; len < shape_dim_output; len++) {
    output_num = output_num * output_shape_[len];
  }

  if (input_num == 0) {
    T *res_arr = static_cast<T *>(malloc(sizeof(T) * output_num));
    for (int res_len = 0; res_len < output_num; res_len++) {
      res_arr[res_len] = static_cast<T>(0);
    }
    auto zerocpret = memcpy_s(output, output_num * sizeof(T), res_arr, output_num * sizeof(T));
    if (zerocpret != EOK) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', memcpy_s error ";
    }
    return true;
  }

  int temp_num = input_num;
  int step = 1;
  T *temp_input = static_cast<T *>(malloc(sizeof(T) * input_num));
  auto cpret = memcpy_s(temp_input, input_num * sizeof(T), input, input_num * sizeof(T));
  if (cpret != EOK) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', memcpy_s error ";
  }
  for (int i = kStridedSliceV2GradMaxInputShapeSize - 1; i >= 0; --i) {
    temp_num = static_cast<int>(temp_num * output_shape_[i] / input_shape_[i]);
    T *temp = static_cast<T *>(malloc(sizeof(T) * temp_num));
    auto res_set = memset_s(temp, sizeof(T) * temp_num, 0, sizeof(T) * temp_num);
    if (res_set != EOK) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', memcpy_s error ";
    }

    int start1 = 0;
    int start2 = 0;
    while (start1 < temp_num) {
      int id = 0;
      for (int k = begin_[i]; strides_[i] > 0 ? k < end_[i] : k > end_[i]; k += strides_[i], id++) {
        auto ret =
          memcpy_s(temp + start1 + k * step, step * sizeof(T), temp_input + start2 + id * step, step * sizeof(T));
        if (ret != EOK) {
          MS_LOG(ERROR) << "For '" << kernel_name_ << "', memcpy_s error ";
        }
      }
      start1 += output_shape_[i] * step;
      start2 += input_shape_[i] * step;
    }
    step *= output_shape_[i];
    temp_input = temp;
  }
  auto res = memcpy_s(output, temp_num * sizeof(T), temp_input, temp_num * sizeof(T));
  if (res != EOK) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', memcpy_s error ";
  }
  free(temp_input);
  return true;
}

int StridedSliceV2GradCpuKernelMod::SignOfStride(size_t axis) const {
  if (strides_[axis] > 0) {
    return 1;
  }
  return -1;
}

void StridedSliceV2GradCpuKernelMod::FormatArgs(bool stride) {
  if (stride) {
    for (size_t i = 0; i < strides_.size(); ++i) {
      if (strides_[i] == 0) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << ", elements in 'stride' can not be 0, but got 0 in dimension "
                          << i;
      }
      if (end_[i] == 0 && begin_[i] < 0) {
        end_[i] = end_[i] + SizeToInt(output_shape_[i]);
      }
      if (end_[i] < 0) {
        end_[i] = end_[i] + SizeToInt(output_shape_[i]) < 0 ? 0 : end_[i] + SizeToInt(output_shape_[i]);
      }
      if (end_[i] > SizeToInt(output_shape_[i])) {
        end_[i] = SizeToInt(output_shape_[i]);
      }
    }
  }
  for (size_t i = 0; i < begin_.size(); i++) {
    if (begin_[i] < 0) {
      auto k = begin_[i] + SizeToInt(output_shape_[i]);
      begin_[i] = k < 0 ? 0 : k;
    }
    if (begin_[i] > SizeToInt(output_shape_[i])) {
      begin_[i] = SizeToInt(output_shape_[i]);
    }
  }
}

std::vector<KernelAttr> StridedSliceV2GradCpuKernelMod::GetOpSupport() {
  auto iter = support_list_map.find(kStridedSliceV2Grad);
  if (iter == support_list_map.end()) {
    MS_LOG(EXCEPTION) << "Does not support StridedSliceV2Grad";
  }

  return iter->second;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, StridedSliceV2Grad, []() {
  return std::make_shared<StridedSliceV2GradCpuKernelMod>(kStridedSliceV2Grad);
});
}  // namespace kernel
}  // namespace mindspore
