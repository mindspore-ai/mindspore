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

#include <algorithm>

#include "plugin/device/gpu/kernel/arrays/strided_slice_grad_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
template <typename T>
using Complex = mindspore::utils::Complex<T>;

bool StridedSliceGradGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For 'StridedSliceGrad', it does not support this kernel type:" << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  auto input_num = inputs.size();
  if (input_num == DynamicInputNum) {
    is_dynamic_attr_ = true;
    return true;
  }
  auto prim = base_operator->GetPrim();
  MS_EXCEPTION_IF_NULL(prim);
  auto begin_value = prim->GetAttr(kAttrBegin);
  MS_EXCEPTION_IF_NULL(begin_value);
  begin_ = GetValue<std::vector<int64_t>>(begin_value);
  auto end_value = prim->GetAttr(kAttrEnd);
  MS_EXCEPTION_IF_NULL(end_value);
  end_ = GetValue<std::vector<int64_t>>(end_value);
  auto strides_value = prim->GetAttr(kAttrStrides);
  MS_EXCEPTION_IF_NULL(strides_value);
  strides_ = GetValue<std::vector<int64_t>>(strides_value);
  auto shapex_value = prim->GetAttr(kAttrShapex);
  MS_EXCEPTION_IF_NULL(shapex_value);
  shapex_ = GetValue<std::vector<int64_t>>(shapex_value);
  return true;
}

int StridedSliceGradGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs,
                                         const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  auto shapex = GetDynamicAttrIntValue(inputs, kShapexIndex_, inputsOnHost, kernel_name_, &shapex_);
  auto begin = GetDynamicAttrIntValue(inputs, kBeginIndex_, inputsOnHost, kernel_name_, &begin_);
  auto end = GetDynamicAttrIntValue(inputs, kEndIndex_, inputsOnHost, kernel_name_, &end_);
  auto stride = GetDynamicAttrIntValue(inputs, kStrideIndex_, inputsOnHost, kernel_name_, &strides_);
  if (shapex && begin && end && stride) {
    get_dynamic_attr_value_ = true;
  }
  auto ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  input_shape_.clear();
  for (auto x : shapex_) {
    input_shape_.push_back(static_cast<size_t>(x));
  }
  if (input_shape_.size() > MAX_DIMS) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be greater than " << MAX_DIMS
                      << ", but got " << input_shape_.size();
  }

  auto shape_tmp = Convert2Long(input_shape_);
  FillEmptyDims(&begin_, &end_, &strides_, &shape_tmp);
  input_shape_ = Convert2SizeT(shape_tmp);
  auto prim = base_operator->GetPrim();
  ComputeBeginMask(&begin_, strides_, shape_tmp, prim);
  ComputeEndMask(&end_, strides_, shape_tmp, prim);
  ComputeEllipsisMask(&begin_, &end_, &strides_, shape_tmp, prim);
  ComputNewAxisMask(&begin_, &end_, &strides_, shape_tmp, prim);
  ComputeShrinkAxisMask(begin_, &end_, &strides_, prim);
  FillOutputDim();
  null_output_ = IsNullOutput();
  return KRET_OK;
}

template <typename T, typename S = int64_t>
bool StridedSliceGradGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                const std::vector<AddressPtr> &outputs) {
  if (is_dynamic_attr_ && !get_dynamic_attr_value_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', fail to get value of dynamic attr!";
  }
  T *dy = GetDeviceAddress<T>(inputs, 0);
  T *dx = GetDeviceAddress<T>(outputs, 0);
  FillDeviceArray(outputs[0]->size / sizeof(T), dx, 0.f, reinterpret_cast<cudaStream_t>(cuda_stream_));
  if (null_output_) {
    return true;
  }

  StridedSliceGrad(output_shape_, begin_, strides_, input_shape_, dy, dx, reinterpret_cast<cudaStream_t>(cuda_stream_));
  return true;
}

std::vector<std::pair<KernelAttr, StridedSliceGradGpuKernelMod::StridedSliceGradLaunchFunc>>
  StridedSliceGradGpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &StridedSliceGradGpuKernelMod::LaunchKernel<double>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &StridedSliceGradGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &StridedSliceGradGpuKernelMod::LaunchKernel<half>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &StridedSliceGradGpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &StridedSliceGradGpuKernelMod::LaunchKernel<int>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
     &StridedSliceGradGpuKernelMod::LaunchKernel<int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
     &StridedSliceGradGpuKernelMod::LaunchKernel<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
     &StridedSliceGradGpuKernelMod::LaunchKernel<uint64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
     &StridedSliceGradGpuKernelMod::LaunchKernel<uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
     &StridedSliceGradGpuKernelMod::LaunchKernel<uint16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
     &StridedSliceGradGpuKernelMod::LaunchKernel<uchar>},
    {KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
     &StridedSliceGradGpuKernelMod::LaunchKernel<bool>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
     &StridedSliceGradGpuKernelMod::LaunchKernel<Complex<float>>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
     &StridedSliceGradGpuKernelMod::LaunchKernel<Complex<double>>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat64),
     &StridedSliceGradGpuKernelMod::LaunchKernel<double, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32),
     &StridedSliceGradGpuKernelMod::LaunchKernel<float, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat16),
     &StridedSliceGradGpuKernelMod::LaunchKernel<float, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64),
     &StridedSliceGradGpuKernelMod::LaunchKernel<int64_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt32),
     &StridedSliceGradGpuKernelMod::LaunchKernel<int, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt16),
     &StridedSliceGradGpuKernelMod::LaunchKernel<int16_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt8),
     &StridedSliceGradGpuKernelMod::LaunchKernel<int8_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt64),
     &StridedSliceGradGpuKernelMod::LaunchKernel<uint64_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt32),
     &StridedSliceGradGpuKernelMod::LaunchKernel<uint32_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt16),
     &StridedSliceGradGpuKernelMod::LaunchKernel<uint16_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt8),
     &StridedSliceGradGpuKernelMod::LaunchKernel<uchar, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeBool),
     &StridedSliceGradGpuKernelMod::LaunchKernel<bool, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeComplex64),
     &StridedSliceGradGpuKernelMod::LaunchKernel<Complex<float>, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeComplex128),
     &StridedSliceGradGpuKernelMod::LaunchKernel<Complex<double>, int64_t>}};

std::vector<KernelAttr> StridedSliceGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, StridedSliceGradGpuKernelMod::StridedSliceGradLaunchFunc> &pair) {
                         return pair.first;
                       });
  return support_list;
}

void StridedSliceGradGpuKernelMod::FillEmptyDims(std::vector<int64_t> *begin, std::vector<int64_t> *end,
                                                 std::vector<int64_t> *stride, ShapeVector *input_shape) {
  std::vector<int64_t> &_begin = *begin;
  std::vector<int64_t> &_end = *end;
  std::vector<int64_t> &_stride = *stride;
  auto &_input_shape = *input_shape;
  if (_begin.size() != _end.size() || _begin.size() != _stride.size() || _begin.size() > _input_shape.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the length of 'begin', 'stride' and 'end' should be equal "
                         "and less than or equal to the dimension of 'input_x', but got the length of 'begin': "
                      << _begin.size() << ", the length of 'stride': " << _stride.size()
                      << ", the length of 'end': " << _end.size()
                      << ", the dimension of 'input_x': " << _input_shape.size();
  }

  for (size_t i = 0; i < kStridedSliceMaxDims; i++) {
    if (i >= _input_shape.size()) {
      _input_shape.push_back(1);
    }

    if (i < _begin.size()) {
      int64_t dim = _input_shape[i];
      _begin[i] = std::min(_begin[i] < 0 ? std::max(_begin[i] + dim, static_cast<int64_t>(0)) : _begin[i], dim - 1);
    } else {
      _begin.push_back(0);
    }

    if (i < _end.size()) {
      int64_t dim = _input_shape[i];
      _end[i] = std::max(_end[i] < 0 ? _end[i] + dim : std::min(_end[i], dim), static_cast<int64_t>(-1));
    } else {
      _end.push_back(i < _input_shape.size() ? _input_shape[i] : 1);
    }

    if (i >= _stride.size()) {
      _stride.push_back(1);
    }
  }
}

void StridedSliceGradGpuKernelMod::ComputeBeginMask(std::vector<int64_t> *begin, const std::vector<int64_t> &stride,
                                                    const ShapeVector &input_shape, const ops::PrimitiveCPtr &prim) {
  auto begin_mask_value = prim->GetAttr(kAttrBeginMask);
  MS_EXCEPTION_IF_NULL(begin_mask_value);
  auto begin_mask_int = GetValue<int64_t>(begin_mask_value);
  std::vector<int64_t> &_begin = *begin;
  auto begin_mask = Dec2Bin(begin_mask_int);
  for (size_t i = 0; i < begin_mask.size(); i++) {
    if (i < kStridedSliceMaxDims && begin_mask[i]) {
      _begin[i] = stride[i] < 0 ? input_shape[i] - 1 : 0;
    }
  }
}

void StridedSliceGradGpuKernelMod::ComputeEndMask(std::vector<int64_t> *end, const std::vector<int64_t> &stride,
                                                  const ShapeVector &input_shape, const ops::PrimitiveCPtr &prim) {
  auto end_mask_value = prim->GetAttr(kAttrEndMask);
  MS_EXCEPTION_IF_NULL(end_mask_value);
  auto end_mask_int = GetValue<int64_t>(end_mask_value);
  std::vector<int64_t> &_end = *end;
  auto end_mask = Dec2Bin(end_mask_int);
  for (size_t j = 0; j < end_mask.size(); j++) {
    if (j < kStridedSliceMaxDims && end_mask[j]) {
      _end[j] = stride[j] < 0 ? -1 : input_shape[j];
    }
  }
}

void StridedSliceGradGpuKernelMod::ComputeEllipsisMask(std::vector<int64_t> *begin, std::vector<int64_t> *end,
                                                       std::vector<int64_t> *stride, const ShapeVector &input_shape,
                                                       const ops::PrimitiveCPtr &prim) {
  auto ellipsis_mask_value = prim->GetAttr(kAttrEllipsisMask);
  MS_EXCEPTION_IF_NULL(ellipsis_mask_value);
  auto ellipsis_mask_int = GetValue<int64_t>(ellipsis_mask_value);
  std::vector<int64_t> &_begin = *begin;
  std::vector<int64_t> &_end = *end;
  std::vector<int64_t> &_stride = *stride;
  auto ellipsis_mask = Dec2Bin(ellipsis_mask_int);
  for (size_t k = 0; k < ellipsis_mask.size(); k++) {
    if (k < kStridedSliceMaxDims && ellipsis_mask[k]) {
      _begin[k] = 0;
      _end[k] = input_shape[k];
      _stride[k] = 1;
    }
  }
}

void StridedSliceGradGpuKernelMod::ComputNewAxisMask(std::vector<int64_t> *begin, std::vector<int64_t> *end,
                                                     std::vector<int64_t> *stride, const ShapeVector &input_shape,
                                                     const ops::PrimitiveCPtr &prim) {
  std::vector<int64_t> &_begin = *begin;
  std::vector<int64_t> &_end = *end;
  std::vector<int64_t> &_stride = *stride;
  auto new_axis_mask_value = prim->GetAttr(kAttrNewAxisMask);
  MS_EXCEPTION_IF_NULL(new_axis_mask_value);
  auto new_axis_mask_int = GetValue<int64_t>(new_axis_mask_value);
  auto new_axis_mask = Dec2Bin(new_axis_mask_int);
  for (size_t l = 0; l < new_axis_mask.size(); l++) {
    if (l < kStridedSliceMaxDims && new_axis_mask[l]) {
      _begin[l] = 0;
      _end[l] = input_shape[l];
      _stride[l] = 1;
    }
  }
}

void StridedSliceGradGpuKernelMod::ComputeShrinkAxisMask(const std::vector<int64_t> &begin, std::vector<int64_t> *end,
                                                         std::vector<int64_t> *stride, const ops::PrimitiveCPtr &prim) {
  auto shrink_axis_mask_value = prim->GetAttr(kAttrShrinkAxisMask);
  MS_EXCEPTION_IF_NULL(shrink_axis_mask_value);
  auto shrink_axis_mask_int = GetValue<int64_t>(shrink_axis_mask_value);
  std::vector<int64_t> &_end = *end;
  std::vector<int64_t> &_stride = *stride;
  auto shrink_axis_mask = Dec2Bin(shrink_axis_mask_int);
  for (size_t m = 0; m < shrink_axis_mask.size(); m++) {
    if (m < kStridedSliceMaxDims && shrink_axis_mask[m]) {
      _end[m] = _end[m] > begin[m] ? begin[m] + 1 : begin[m] - 1;
      _stride[m] = _end[m] > begin[m] ? 1 : -1;
    }
  }
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, StridedSliceGrad, StridedSliceGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
