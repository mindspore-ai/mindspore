/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/mkldnn/conv_grad_filter_cpu_kernel.h"

#include <string>
#include <algorithm>
#include <map>
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kConv2DBackpropFilter = "Conv2DBackpropFilter";
constexpr auto kConv3DBackpropFilter = "Conv3DBackpropFilter";
constexpr size_t kConvGradFilterInputsMinNum = 2;
constexpr size_t kConvGradFilterOutputsNum = 1;
}  // namespace

bool ConvGradFilterCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  if (kernel_name_ == kConv2DBackpropFilterOpName) {
    src_index_ = 1;
    diff_dst_index_ = 0;
  }
  auto prim = base_operator->GetPrim();
  MS_EXCEPTION_IF_NULL(prim);
  format_ = GetValue<std::string>(prim->GetAttr(FORMAT));
  group_ = GetValue<int64_t>(prim->GetAttr(GROUP));
  pad_mode_ = GetValue<std::string>(prim->GetAttr(PAD_MODE));
  if (format_ != NCHW && format_ != NCDHW) {
    MS_LOG(EXCEPTION) << kernel_name_ << " only supports " << NCHW << " or " << NCDHW << " format "
                      << ", but got format: " << format_;
  }
  const auto stride_attr = format_ == NCHW ? STRIDE : STRIDES;
  const auto dilation_attr = format_ == NCHW ? DILATION : DILATIONS;
  strides_include_nc_ = GetValue<std::vector<int64_t>>(prim->GetAttr(stride_attr));
  dilation_include_nc_ = GetValue<std::vector<int64_t>>(prim->GetAttr(dilation_attr));
  return true;
}

int ConvGradFilterCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs,
                                       const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto src_shape = inputs[src_index_]->GetDeviceShapeAdaptively();
  auto dst_shape = inputs[diff_dst_index_]->GetDeviceShapeAdaptively();
  auto weight_shape = outputs[0]->GetDeviceShapeAdaptively();
  size_t src_dim = src_shape.size();
  if (src_dim != weight_shape.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the rank of input must be euqal to weight's, but got input shape: " << src_shape
                      << ", weight shape: " << weight_shape;
  }
  if (src_dim != SHAPE_4D && src_dim != SHAPE_5D) {
    MS_LOG(EXCEPTION) << "Conv Grad only supports 4D/5D input, but got " << src_dim << "D!";
  }
  const auto &format = format_;
  if (src_dim == SHAPE_4D && format != NCHW) {
    MS_LOG(EXCEPTION) << kernel_name_ << " only supports 4D input with NCHW format, but got format " << format;
  }
  if (src_dim == SHAPE_5D && format != NCDHW) {
    MS_LOG(EXCEPTION) << kernel_name_ << " only supports 5D input with NCDHW format, but got fornat " << format;
  }
  dnnl::memory::dims kernel_size(weight_shape.begin() + NC_LEN, weight_shape.end());
  const auto group = group_;
  if (group > 1) {
    if (src_shape[1] % group != 0) {
      MS_LOG(EXCEPTION) << kernel_name_ << " requires channels must be divided by group!";
    }
    (void)weight_shape.insert(weight_shape.begin(), group);
    weight_shape[1] = weight_shape[1] / group;
  }
  const dnnl::memory::desc src_desc = GetDefaultMemDesc(src_shape);
  const dnnl::memory::desc weights_desc = GetDefaultMemDesc(weight_shape);
  const dnnl::memory::desc dst_desc = GetDefaultMemDesc(dst_shape);
  const auto &strides_include_nc = strides_include_nc_;
  const auto &dilation_include_nc = dilation_include_nc_;
  if (strides_include_nc.size() != src_dim) {
    MS_LOG(EXCEPTION) << kernel_name_ << "requires strides must be " << src_dim << "D, but got "
                      << strides_include_nc.size() << "D!";
  }
  if (dilation_include_nc.size() != src_dim) {
    MS_LOG(EXCEPTION) << kernel_name_ << " requires dilation must be " << src_dim << "D, but got "
                      << dilation_include_nc.size() << "D!";
  }
  const dnnl::memory::dims strides(strides_include_nc.begin() + NC_LEN, strides_include_nc.end());
  const dnnl::memory::dims dilation(dilation_include_nc.begin() + NC_LEN, dilation_include_nc.end());
  dnnl::memory::dims dilates;
  dnnl::memory::dims padding_l;
  dnnl::memory::dims padding_r;
  (void)std::transform(dilation.begin(), dilation.end(), std::back_inserter(dilates),
                       [](const int64_t &value) { return value - 1; });
  const auto &pad_mode = pad_mode_;
  PaddingInfo padding_info{pad_mode, kernel_size, strides, dilation, &padding_l, &padding_r};
  GetPadding(base_operator, src_shape, padding_info);
  const auto forward_desc = CreateDesc<dnnl::convolution_forward::desc>(
    dnnl::prop_kind::forward_training, dnnl::algorithm::convolution_auto, src_desc, weights_desc, dst_desc, strides,
    dilates, padding_l, padding_r);
  const auto forward_prim_desc = CreateDesc<dnnl::convolution_forward::primitive_desc>(forward_desc, engine_);
  const auto backward_desc = CreateDesc<dnnl::convolution_backward_weights::desc>(
    dnnl::algorithm::convolution_auto, src_desc, weights_desc, dst_desc, strides, dilates, padding_l, padding_r);
  const auto backward_prim_desc =
    CreateDesc<dnnl::convolution_backward_weights::primitive_desc>(backward_desc, engine_, forward_prim_desc);
  primitive_ = CreatePrimitive<dnnl::convolution_backward_weights>(backward_prim_desc);
  AddArgument(DNNL_ARG_SRC, src_desc);
  AddArgument(DNNL_ARG_DIFF_DST, dst_desc);
  AddArgument(DNNL_ARG_DIFF_WEIGHTS, weights_desc);
  return KRET_OK;
}

bool ConvGradFilterCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.size() < kConvGradFilterInputsMinNum) {
    MS_LOG(EXCEPTION) << "Input numbers can not less " << kConvGradFilterInputsMinNum << ", but got " << inputs.size();
  }
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kConvGradFilterOutputsNum, kernel_name_);
  SetArgumentHandle(DNNL_ARG_SRC, inputs[src_index_]->addr);
  SetArgumentHandle(DNNL_ARG_DIFF_DST, inputs[diff_dst_index_]->addr);
  SetArgumentHandle(DNNL_ARG_DIFF_WEIGHTS, outputs[0]->addr);
  ExecutePrimitive();
  return true;
}

std::vector<KernelAttr> ConvGradFilterCpuKernelMod::GetOpSupport() {
  static std::map<std::string, std::vector<KernelAttr>> support_list_map = {
    {kConv2DBackpropFilter,
     {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
      KernelAttr()
        .AddInputAttr(kNumberTypeFloat32)
        .AddInputAttr(kNumberTypeFloat32)
        .AddInputAttr(kNumberTypeInt64)
        .AddOutputAttr(kNumberTypeFloat32),
      KernelAttr()
        .AddInputAttr(kNumberTypeFloat32)
        .AddInputAttr(kNumberTypeFloat32)
        .AddInputAttr(kNumberTypeInt32)
        .AddOutputAttr(kNumberTypeFloat32)}},
    {kConv3DBackpropFilter,
     {KernelAttr()
        .AddInputAttr(kNumberTypeFloat32)
        .AddInputAttr(kNumberTypeFloat32)
        .AddInputAttr(kNumberTypeInt64)
        .AddOutputAttr(kNumberTypeFloat32)}}};
  auto iter = support_list_map.find(kernel_type_);
  if (iter == support_list_map.end()) {
    MS_LOG(EXCEPTION) << "ConvGradFilter does not support " << kernel_type_;
  }
  return iter->second;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Conv2DBackpropFilter,
                                 []() { return std::make_shared<ConvGradFilterCpuKernelMod>(kConv2DBackpropFilter); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Conv3DBackpropFilter,
                                 []() { return std::make_shared<ConvGradFilterCpuKernelMod>(kConv3DBackpropFilter); });
}  // namespace kernel
}  // namespace mindspore
