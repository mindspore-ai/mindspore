/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include <memory>
#include "plugin/device/gpu/kernel/nn/maxpool_with_argmax_v2_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/maxpool_with_argmax_v2_impl.cuh"
#include "mindspore/core/ops/max_pool_with_argmax_v2.h"

namespace mindspore {
namespace kernel {
constexpr auto kMaxPoolWithArgmaxV2 = "MaxPoolWithArgmaxV2";
constexpr size_t kInputDimLowerLimit = 4;
constexpr size_t kOutputDimLowerLimit = 4;
constexpr size_t kInputNum = 1;
constexpr size_t kOutputNum = 2;
const size_t DIM_SIZE_1 = 1;
const size_t DIM_SIZE_4 = 4;

template <typename T, typename S>
bool MaxPoolWithArgmaxV2FwdGpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                      const std::vector<kernel::AddressPtr> &outputs) {
  if (is_null_input_) {
    return true;
  }
  T *input_addr = GetDeviceAddress<T>(inputs, 0);
  T *output_addr = GetDeviceAddress<T>(outputs, 0);
  S *index_addr = GetDeviceAddress<S>(outputs, 1);

  CalMaxPoolWithArgmaxV2(input_addr, in_n_, in_c_, in_h_, in_w_, ksize_h_, ksize_w_, strides_h_, strides_w_, pads_h_,
                         pads_w_, dilation_h_, dilation_w_, out_h_, out_w_, output_addr, index_addr, device_id_,
                         reinterpret_cast<cudaStream_t>(cuda_stream_));
  return true;
}

std::vector<int> GetAttrFromOpsPrim(const std::vector<int64_t> &attr) {
  if (attr.size() == DIM_SIZE_1) {
    return {LongToInt(attr[kIndex0]), LongToInt(attr[kIndex0])};
  } else if (attr.size() == DIM_SIZE_4) {
    return {LongToInt(attr[kIndex2]), LongToInt(attr[kIndex3])};
  } else {
    return {LongToInt(attr[kIndex0]), LongToInt(attr[kIndex1])};
  }
}

bool MaxPoolWithArgmaxV2FwdGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                              const std::vector<KernelTensorPtr> &inputs,
                                              const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  auto kernel_ptr = std::make_shared<ops::MaxPoolWithArgmaxV2>(base_operator->GetPrim());
  auto ksize = kernel_ptr->get_kernel_size();
  auto strides = kernel_ptr->get_strides();
  auto pads = kernel_ptr->get_pads();
  auto dilation = kernel_ptr->get_dilation();

  auto ksize_v = GetAttrFromOpsPrim(ksize);
  ksize_h_ = ksize_v[0];
  ksize_w_ = ksize_v[1];

  auto strides_v = GetAttrFromOpsPrim(strides);
  strides_h_ = strides_v[0];
  strides_w_ = strides_v[1];

  auto pads_v = GetAttrFromOpsPrim(pads);
  pads_h_ = pads_v[0];
  pads_w_ = pads_v[1];

  auto dilation_v = GetAttrFromOpsPrim(dilation);
  dilation_h_ = dilation_v[0];
  dilation_w_ = dilation_v[1];

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' it does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int MaxPoolWithArgmaxV2FwdGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                               const std::vector<KernelTensorPtr> &inputs,
                                               const std::vector<KernelTensorPtr> &outputs,
                                               const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
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
  if ((input_shape.size() < kInputDimLowerLimit) || (output_shape.size() < kOutputDimLowerLimit)) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of input and output cannot be less than "
                  << kOutputDimLowerLimit << ", but got the dimension of input: " << input_shape.size()
                  << ", the dimension of output: " << output_shape.size();
    return KRET_RESIZE_FAILED;
  }

  in_n_ = LongToInt(input_shape[kIndex0]);
  in_c_ = LongToInt(input_shape[kIndex1]);
  in_h_ = LongToInt(input_shape[kIndex2]);
  in_w_ = LongToInt(input_shape[kIndex3]);

  out_h_ = LongToInt(output_shape[kIndex2]);
  out_w_ = LongToInt(output_shape[kIndex3]);

  return KRET_OK;
}

std::vector<std::pair<KernelAttr, MaxPoolWithArgmaxV2FwdGpuKernelMod::MaxPoolWithArgmaxV2Func>>
  MaxPoolWithArgmaxV2FwdGpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32),
     &MaxPoolWithArgmaxV2FwdGpuKernelMod::LaunchKernel<half, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
     &MaxPoolWithArgmaxV2FwdGpuKernelMod::LaunchKernel<float, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt32),
     &MaxPoolWithArgmaxV2FwdGpuKernelMod::LaunchKernel<double, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt32),
     &MaxPoolWithArgmaxV2FwdGpuKernelMod::LaunchKernel<int8_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt32),
     &MaxPoolWithArgmaxV2FwdGpuKernelMod::LaunchKernel<int16_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &MaxPoolWithArgmaxV2FwdGpuKernelMod::LaunchKernel<int32_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
     &MaxPoolWithArgmaxV2FwdGpuKernelMod::LaunchKernel<int64_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt32),
     &MaxPoolWithArgmaxV2FwdGpuKernelMod::LaunchKernel<uint8_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt32),
     &MaxPoolWithArgmaxV2FwdGpuKernelMod::LaunchKernel<uint16_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt32),
     &MaxPoolWithArgmaxV2FwdGpuKernelMod::LaunchKernel<uint32_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt32),
     &MaxPoolWithArgmaxV2FwdGpuKernelMod::LaunchKernel<uint64_t, int32_t>},

    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt64),
     &MaxPoolWithArgmaxV2FwdGpuKernelMod::LaunchKernel<half, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt64),
     &MaxPoolWithArgmaxV2FwdGpuKernelMod::LaunchKernel<float, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt64),
     &MaxPoolWithArgmaxV2FwdGpuKernelMod::LaunchKernel<double, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt64),
     &MaxPoolWithArgmaxV2FwdGpuKernelMod::LaunchKernel<int8_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt64),
     &MaxPoolWithArgmaxV2FwdGpuKernelMod::LaunchKernel<int16_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
     &MaxPoolWithArgmaxV2FwdGpuKernelMod::LaunchKernel<int32_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &MaxPoolWithArgmaxV2FwdGpuKernelMod::LaunchKernel<int64_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt64),
     &MaxPoolWithArgmaxV2FwdGpuKernelMod::LaunchKernel<uint8_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt64),
     &MaxPoolWithArgmaxV2FwdGpuKernelMod::LaunchKernel<uint16_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt64),
     &MaxPoolWithArgmaxV2FwdGpuKernelMod::LaunchKernel<uint32_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt64),
     &MaxPoolWithArgmaxV2FwdGpuKernelMod::LaunchKernel<uint64_t, int64_t>},
};

std::vector<KernelAttr> MaxPoolWithArgmaxV2FwdGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MaxPoolWithArgmaxV2Func> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, MaxPoolWithArgmaxV2, []() {
  return std::make_shared<MaxPoolWithArgmaxV2FwdGpuKernelMod>(kMaxPoolWithArgmaxV2);
});
}  // namespace kernel
}  // namespace mindspore
