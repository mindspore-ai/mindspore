/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/nn/maxpool3d_with_argmax_gpu_kernel.h"
#include <algorithm>
#include <memory>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/maxpool3d_with_argmax_impl.cuh"
#include "mindspore/core/ops/max_pool3d_with_argmax.h"

namespace mindspore {
namespace kernel {
constexpr auto kMaxPool3DWithArgmax = "MaxPool3DWithArgmax";
constexpr size_t kInputDimLowerLimit = 5;
constexpr size_t kOutputDimLowerLimit = 5;
constexpr size_t kInputNum = 1;
constexpr size_t kOutputNum = 2;
constexpr size_t kMinAttrSize = 3;
constexpr int kIndexRevrseW = -1;
constexpr int kIndexRevrseH = -2;
constexpr int kIndexRevrseD = -3;

template <typename T, typename S>
bool MaxPool3DWithArgmaxFwdGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                      const std::vector<AddressPtr> &outputs) {
  if (is_null_input_) {
    return true;
  }
  T *input_addr = GetDeviceAddress<T>(inputs, 0);
  T *output_addr = GetDeviceAddress<T>(outputs, 0);
  S *index_addr = GetDeviceAddress<S>(outputs, 1);

  CalMaxPool3DWithArgmax(input_addr, in_n_, in_c_, in_d_, in_h_, in_w_, ksize_d_, ksize_h_, ksize_w_, stride_d_,
                         stride_h_, stride_w_, pad_d_, pad_h_, pad_w_, dilation_d_, dilation_h_, dilation_w_, out_d_,
                         out_h_, out_w_, output_addr, index_addr, device_id_,
                         reinterpret_cast<cudaStream_t>(cuda_stream_));
  return true;
}

bool MaxPool3DWithArgmaxFwdGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                              const std::vector<KernelTensorPtr> &inputs,
                                              const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  auto kernel_ptr = std::make_shared<ops::MaxPool3DWithArgmax>(base_operator->GetPrim());
  auto ksize = kernel_ptr->get_kernel_size();
  auto strides = kernel_ptr->get_strides();
  auto pads = kernel_ptr->get_pads();
  auto dilation = kernel_ptr->get_dilation();
  if (ksize.size() < kMinAttrSize) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the attr 'ksize' dims should not less than " << kMinAttrSize
                  << ", but got " << ksize.size();
    return false;
  }
  if (strides.size() < kMinAttrSize) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the attr 'strides' dims should not less than " << kMinAttrSize
                  << ", but got " << strides.size();
    return false;
  }
  if (pads.size() < kMinAttrSize) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the attr 'pads' dims should not less than " << kMinAttrSize
                  << ", but got " << pads.size();
    return false;
  }
  if (dilation.size() < kMinAttrSize) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the attr 'dilation' dims should not less than " << kMinAttrSize
                  << ", but got " << dilation.size();
    return false;
  }

  ksize_d_ = LongToInt(ksize[kIndex0]);
  ksize_h_ = LongToInt(ksize[kIndex1]);
  ksize_w_ = LongToInt(ksize[kIndex2]);

  stride_d_ = LongToInt(strides[kIndex0]);
  stride_h_ = LongToInt(strides[kIndex1]);
  stride_w_ = LongToInt(strides[kIndex2]);

  pad_d_ = LongToInt(pads[kIndex0]);
  pad_h_ = LongToInt(pads[kIndex1]);
  pad_w_ = LongToInt(pads[kIndex2]);

  dilation_d_ = LongToInt(dilation[kIndex0]);
  dilation_h_ = LongToInt(dilation[kIndex1]);
  dilation_w_ = LongToInt(dilation[kIndex2]);

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' it does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int MaxPool3DWithArgmaxFwdGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                               const std::vector<KernelTensorPtr> &inputs,
                                               const std::vector<KernelTensorPtr> &outputs,
                                               const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  if (inputs.size() != kInputNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the number of inputs should be " << kInputNum << ", but got "
                  << inputs.size();
    return KRET_RESIZE_FAILED;
  }
  if (outputs.size() != kOutputNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the number of outputs should be " << kOutputNum << ", but got "
                  << outputs.size();
    return KRET_RESIZE_FAILED;
  }
  auto input_shape = inputs.at(kIndex0)->GetShapeVector();
  auto output_shape = outputs.at(kIndex0)->GetShapeVector();
  is_null_input_ =
    CHECK_SHAPE_NULL(input_shape, kernel_name_, "input") || CHECK_SHAPE_NULL(output_shape, kernel_name_, "output");
  if (is_null_input_) {
    return KRET_RESIZE_FAILED;
  }
  if (input_shape.size() < kInputDimLowerLimit || output_shape.size() < kOutputDimLowerLimit) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of input and output cannot be less than 4, but "
                  << "got the dimension of input: " << input_shape.size()
                  << ", the dimension of output: " << output_shape.size();
    return KRET_RESIZE_FAILED;
  }

  in_n_ = LongToInt(input_shape[kIndex0]);
  in_c_ = LongToInt(input_shape[kIndex1]);
  in_d_ = LongToInt(input_shape[kIndex2]);
  in_h_ = LongToInt(input_shape[kIndex3]);
  in_w_ = LongToInt(input_shape[kIndex4]);

  out_d_ = LongToInt(output_shape[kIndex2]);
  out_h_ = LongToInt(output_shape[kIndex3]);
  out_w_ = LongToInt(output_shape[kIndex4]);

  return KRET_OK;
}

std::vector<std::pair<KernelAttr, MaxPool3DWithArgmaxFwdGpuKernelMod::MaxPool3DArgMaxFunc>>
  MaxPool3DWithArgmaxFwdGpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32),
     &MaxPool3DWithArgmaxFwdGpuKernelMod::LaunchKernel<half, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
     &MaxPool3DWithArgmaxFwdGpuKernelMod::LaunchKernel<float, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt32),
     &MaxPool3DWithArgmaxFwdGpuKernelMod::LaunchKernel<double, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt32),
     &MaxPool3DWithArgmaxFwdGpuKernelMod::LaunchKernel<int8_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt32),
     &MaxPool3DWithArgmaxFwdGpuKernelMod::LaunchKernel<int16_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &MaxPool3DWithArgmaxFwdGpuKernelMod::LaunchKernel<int32_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
     &MaxPool3DWithArgmaxFwdGpuKernelMod::LaunchKernel<int64_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt32),
     &MaxPool3DWithArgmaxFwdGpuKernelMod::LaunchKernel<uint8_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt32),
     &MaxPool3DWithArgmaxFwdGpuKernelMod::LaunchKernel<uint16_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt32),
     &MaxPool3DWithArgmaxFwdGpuKernelMod::LaunchKernel<uint32_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt32),
     &MaxPool3DWithArgmaxFwdGpuKernelMod::LaunchKernel<uint64_t, int32_t>},

    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt64),
     &MaxPool3DWithArgmaxFwdGpuKernelMod::LaunchKernel<half, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt64),
     &MaxPool3DWithArgmaxFwdGpuKernelMod::LaunchKernel<float, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt64),
     &MaxPool3DWithArgmaxFwdGpuKernelMod::LaunchKernel<double, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt64),
     &MaxPool3DWithArgmaxFwdGpuKernelMod::LaunchKernel<int8_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt64),
     &MaxPool3DWithArgmaxFwdGpuKernelMod::LaunchKernel<int16_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
     &MaxPool3DWithArgmaxFwdGpuKernelMod::LaunchKernel<int32_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &MaxPool3DWithArgmaxFwdGpuKernelMod::LaunchKernel<int64_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt64),
     &MaxPool3DWithArgmaxFwdGpuKernelMod::LaunchKernel<uint8_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt64),
     &MaxPool3DWithArgmaxFwdGpuKernelMod::LaunchKernel<uint16_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt64),
     &MaxPool3DWithArgmaxFwdGpuKernelMod::LaunchKernel<uint32_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt64),
     &MaxPool3DWithArgmaxFwdGpuKernelMod::LaunchKernel<uint64_t, int64_t>},
};

std::vector<KernelAttr> MaxPool3DWithArgmaxFwdGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MaxPool3DArgMaxFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, MaxPool3DWithArgmax, []() {
  return std::make_shared<MaxPool3DWithArgmaxFwdGpuKernelMod>(kMaxPool3DWithArgmax);
});
}  // namespace kernel
}  // namespace mindspore
