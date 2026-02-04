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
#include "kernel/gpu/cuda/arrays/strided_slice_grad_gpu_kernel.h"

#include <algorithm>

#include "include/runtime/hardware_abstract/kernel_base/kernel_utils.h"
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
template <typename T>
using Complex = mindspore::utils::Complex<T>;

bool StridedSliceGradGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For 'StridedSliceGrad', it does not support this kernel type:" << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int StridedSliceGradGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  auto ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  begin_ = inputs[kBeginIndex_]->GetValueWithCheck<std::vector<int64_t>>();
  end_ = inputs[kEndIndex_]->GetValueWithCheck<std::vector<int64_t>>();
  strides_ = inputs[kStrideIndex_]->GetValueWithCheck<std::vector<int64_t>>();
  shapex_ = inputs[kShapexIndex_]->GetValueWithCheck<std::vector<int64_t>>();
  input_shape_.clear();
  for (auto x : shapex_) {
    input_shape_.push_back(static_cast<size_t>(x));
  }
  if (input_shape_.size() > MAX_DIMS) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be greater than " << MAX_DIMS
                      << ", but got " << input_shape_.size();
  }
  auto shape_tmp = Convert2Long(input_shape_);
  FillEmptyDims(kernel_name_, &begin_, &end_, &strides_, &shape_tmp);
  input_shape_ = Convert2SizeT(shape_tmp);
  ComputeBeginMask(&begin_, strides_, shape_tmp, primitive_);
  ComputeEndMask(&end_, strides_, shape_tmp, primitive_);
  ComputeEllipsisMask(&begin_, &end_, &strides_, shape_tmp, primitive_);
  ComputNewAxisMask(&begin_, &end_, &strides_, shape_tmp, primitive_);
  ComputeShrinkAxisMask(begin_, &end_, &strides_, primitive_);
  FillOutputDim();
  null_output_ = IsNullOutput();
  return KRET_OK;
}

template <typename T, typename S = int64_t>
bool StridedSliceGradGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                const std::vector<KernelTensor *> &outputs) {
  if (IsEmptyInput(inputs[0]->size())) {
    return true;
  }
  T *dy = GetDeviceAddress<T>(inputs, 0);
  T *dx = GetDeviceAddress<T>(outputs, 0);
  auto status = FillDeviceArray(outputs[0]->size() / sizeof(T), dx, 0.f, reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  if (null_output_) {
    return true;
  }

  status = StridedSliceGrad(output_shape_, begin_, strides_, input_shape_, dy, dx,
                            reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, StridedSliceGradGpuKernelMod::StridedSliceGradLaunchFunc>>
  StridedSliceGradGpuKernelMod::func_list_ = {{KernelAttr()
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
                                               &StridedSliceGradGpuKernelMod::LaunchKernel<half, int64_t>},
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
                                               &StridedSliceGradGpuKernelMod::LaunchKernel<Complex<double>, int64_t>},
                                              {KernelAttr()
                                                 .AddInputAttr(kNumberTypeFloat64)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddOutputAttr(kNumberTypeFloat64),
                                               &StridedSliceGradGpuKernelMod::LaunchKernel<double, int32_t>},
                                              {KernelAttr()
                                                 .AddInputAttr(kNumberTypeFloat32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddOutputAttr(kNumberTypeFloat32),
                                               &StridedSliceGradGpuKernelMod::LaunchKernel<float, int32_t>},
                                              {KernelAttr()
                                                 .AddInputAttr(kNumberTypeFloat16)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddOutputAttr(kNumberTypeFloat16),
                                               &StridedSliceGradGpuKernelMod::LaunchKernel<half, int32_t>},
                                              {KernelAttr()
                                                 .AddInputAttr(kNumberTypeInt64)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddOutputAttr(kNumberTypeInt32),
                                               &StridedSliceGradGpuKernelMod::LaunchKernel<int64_t, int32_t>},
                                              {KernelAttr()
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddOutputAttr(kNumberTypeInt32),
                                               &StridedSliceGradGpuKernelMod::LaunchKernel<int, int32_t>},
                                              {KernelAttr()
                                                 .AddInputAttr(kNumberTypeInt16)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddOutputAttr(kNumberTypeInt16),
                                               &StridedSliceGradGpuKernelMod::LaunchKernel<int16_t, int32_t>},
                                              {KernelAttr()
                                                 .AddInputAttr(kNumberTypeInt8)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddOutputAttr(kNumberTypeInt8),
                                               &StridedSliceGradGpuKernelMod::LaunchKernel<int8_t, int32_t>},
                                              {KernelAttr()
                                                 .AddInputAttr(kNumberTypeUInt64)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddOutputAttr(kNumberTypeUInt64),
                                               &StridedSliceGradGpuKernelMod::LaunchKernel<uint64_t, int32_t>},
                                              {KernelAttr()
                                                 .AddInputAttr(kNumberTypeUInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddOutputAttr(kNumberTypeUInt32),
                                               &StridedSliceGradGpuKernelMod::LaunchKernel<uint32_t, int32_t>},
                                              {KernelAttr()
                                                 .AddInputAttr(kNumberTypeUInt16)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddOutputAttr(kNumberTypeUInt16),
                                               &StridedSliceGradGpuKernelMod::LaunchKernel<uint16_t, int32_t>},
                                              {KernelAttr()
                                                 .AddInputAttr(kNumberTypeUInt8)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddOutputAttr(kNumberTypeUInt8),
                                               &StridedSliceGradGpuKernelMod::LaunchKernel<uchar, int32_t>},
                                              {KernelAttr()
                                                 .AddInputAttr(kNumberTypeBool)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddOutputAttr(kNumberTypeBool),
                                               &StridedSliceGradGpuKernelMod::LaunchKernel<bool, int32_t>},
                                              {KernelAttr()
                                                 .AddInputAttr(kNumberTypeComplex64)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddOutputAttr(kNumberTypeComplex64),
                                               &StridedSliceGradGpuKernelMod::LaunchKernel<Complex<float>, int32_t>},
                                              {KernelAttr()
                                                 .AddInputAttr(kNumberTypeComplex128)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddInputAttr(kNumberTypeInt32)
                                                 .AddOutputAttr(kNumberTypeComplex128),
                                               &StridedSliceGradGpuKernelMod::LaunchKernel<Complex<double>, int32_t>}};

std::vector<KernelAttr> StridedSliceGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, StridedSliceGradGpuKernelMod::StridedSliceGradLaunchFunc> &pair) {
                         return pair.first;
                       });
  return support_list;
}

void StridedSliceGradGpuKernelMod::FillEmptyDims(const std::string &kernel_name, std::vector<int64_t> *begin,
                                                 std::vector<int64_t> *end, std::vector<int64_t> *stride,
                                                 ShapeVector *input_shape) {
  std::vector<int64_t> &_begin = *begin;
  std::vector<int64_t> &_end = *end;
  std::vector<int64_t> &_stride = *stride;
  auto &_input_shape = *input_shape;
  if (_begin.size() != _end.size() || _begin.size() != _stride.size() || _begin.size() > _input_shape.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name
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
                                                    const ShapeVector &input_shape, const PrimitivePtr &primitive_) {
  auto begin_mask_value = primitive_->GetAttr("begin_mask");
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
                                                  const ShapeVector &input_shape, const PrimitivePtr &primitive_) {
  auto end_mask_value = primitive_->GetAttr("end_mask");
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
                                                       const PrimitivePtr &primitive_) {
  auto ellipsis_mask_value = primitive_->GetAttr("ellipsis_mask");
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
                                                     const PrimitivePtr &primitive_) {
  std::vector<int64_t> &_begin = *begin;
  std::vector<int64_t> &_end = *end;
  std::vector<int64_t> &_stride = *stride;
  auto new_axis_mask_value = primitive_->GetAttr("new_axis_mask");
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
                                                         std::vector<int64_t> *stride, const PrimitivePtr &primitive_) {
  auto shrink_axis_mask_value = primitive_->GetAttr("shrink_axis_mask");
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
