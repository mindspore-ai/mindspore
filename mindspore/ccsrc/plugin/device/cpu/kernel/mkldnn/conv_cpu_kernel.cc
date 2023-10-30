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

#include "plugin/device/cpu/kernel/mkldnn/conv_cpu_kernel.h"
#include <map>
#include <string>
#include <algorithm>

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kConvInputsNum = 2;
constexpr size_t kConvOutputsNum = 1;
}  // namespace

bool ConvCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  format_ = GetValue<std::string>(KernelMod::primitive_->GetAttr(kAttrFormat));
  group_ = GetValue<int64_t>(KernelMod::primitive_->GetAttr(kAttrGroup));
  auto pad_mode_str = GetValue<std::string>(KernelMod::primitive_->GetAttr(kAttrPadMode));
  std::map<std::string, mindspore::PadMode> str2padmode_map = {
    {PAD_MODE_LOWER_SAME, PadMode::SAME},   {PAD_MODE_UPPER_SAME, PadMode::SAME},
    {PAD_MODE_LOWER_VALID, PadMode::VALID}, {PAD_MODE_UPPER_VALID, PadMode::VALID},
    {PAD_MODE_LOWER_PAD, PadMode::PAD},     {PAD_MODE_UPPER_PAD, PadMode::PAD}};
  auto iter = str2padmode_map.find(pad_mode_str);
  if (iter == str2padmode_map.end()) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", pad_mode is illegal, got " << pad_mode_str;
  } else {
    pad_mode_ = iter->second;
  }
  return true;
}

int ConvCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  auto src_shape = inputs[kIndex0]->GetDeviceShapeVector();
  auto weight_shape = inputs[kIndex1]->GetDeviceShapeVector();
  auto dst_shape = outputs[kIndex0]->GetDeviceShapeVector();
  auto src_dim = src_shape.size();
  if (src_dim != weight_shape.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the rank of input must be euqal to weight's, but got input shape: " << src_shape
                      << ", weight shape: " << weight_shape;
  }
  if (src_dim == SHAPE_4D && format_ != NCHW) {
    MS_LOG(ERROR) << kernel_name_ << " only supports 4D input with format NCHW, but got format " << format_;
    return KRET_RESIZE_FAILED;
  }
  if (src_dim == SHAPE_5D && format_ != NCDHW) {
    MS_LOG(ERROR) << kernel_name_ << " only supports 5D input with format NCDHW, but got format " << format_;
    return KRET_RESIZE_FAILED;
  }
  dnnl::memory::dims kernel_size(weight_shape.begin() + NC_LEN, weight_shape.end());
  if (group_ > 1) {
    if (src_shape[1] % group_ != 0) {
      MS_LOG(ERROR) << kernel_name_ << " requires channels must be divided by group!";
      return KRET_RESIZE_FAILED;
    }
    (void)weight_shape.insert(weight_shape.begin(), group_);
    weight_shape[1] = weight_shape[1] / group_;
  }

  const dnnl::memory::desc src_desc = GetDefaultMemDesc(src_shape);
  const dnnl::memory::desc weights_desc = GetDefaultMemDesc(weight_shape);
  const dnnl::memory::desc dst_desc = GetDefaultMemDesc(dst_shape);
  const auto stride_attr = src_dim == SHAPE_4D ? STRIDE : STRIDES;
  const auto dilation_attr = src_dim == SHAPE_4D ? DILATION : DILATIONS;
  const auto strides_include_nc = GetValue<std::vector<int64_t>>(KernelMod::primitive_->GetAttr(stride_attr));
  const auto dilation_include_nc = GetValue<std::vector<int64_t>>(KernelMod::primitive_->GetAttr(dilation_attr));
  if (strides_include_nc.size() != src_dim) {
    MS_LOG(ERROR) << kernel_name_ << "requires strides must be " << src_dim << "D, but got "
                  << strides_include_nc.size() << "D!";
    return KRET_RESIZE_FAILED;
  }
  if (dilation_include_nc.size() != src_dim) {
    MS_LOG(ERROR) << kernel_name_ << " requires dilation must be " << src_dim << "D, but got "
                  << dilation_include_nc.size() << "D!";
    return KRET_RESIZE_FAILED;
  }
  const dnnl::memory::dims strides(strides_include_nc.begin() + NC_LEN, strides_include_nc.end());
  const dnnl::memory::dims dilation(dilation_include_nc.begin() + NC_LEN, dilation_include_nc.end());
  dnnl::memory::dims dilates;
  dnnl::memory::dims padding_l;
  dnnl::memory::dims padding_r;
  (void)std::transform(dilation.begin(), dilation.end(), std::back_inserter(dilates),
                       [](const int64_t &value) { return value - 1; });
  PaddingInfo padding_info{pad_mode_, kernel_size, strides, dilation, &padding_l, &padding_r};
  GetPadding(src_shape, padding_info);

  const auto desc = CreateDesc<dnnl::convolution_forward::desc>(
    dnnl::prop_kind::forward_training, dnnl::algorithm::convolution_auto, src_desc, weights_desc, dst_desc, strides,
    dilates, padding_l, padding_r);
  const auto prim_desc = CreateDesc<dnnl::convolution_forward::primitive_desc>(desc, engine_);
  primitive_ = CreatePrimitive<dnnl::convolution_forward>(prim_desc);
  AddArgument(DNNL_ARG_SRC, src_desc);
  AddArgument(DNNL_ARG_WEIGHTS, weights_desc);
  AddArgument(DNNL_ARG_DST, dst_desc);
  return KRET_OK;
}

bool ConvCpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                              const std::vector<kernel::KernelTensor *> &,
                              const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kConvInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kConvOutputsNum, kernel_name_);
  SetArgumentHandle(DNNL_ARG_SRC, inputs[0]->device_ptr());
  SetArgumentHandle(DNNL_ARG_WEIGHTS, inputs[1]->device_ptr());
  SetArgumentHandle(DNNL_ARG_DST, outputs[0]->device_ptr());
  ExecutePrimitive();
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Conv2D, ConvCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Conv3D, ConvCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
