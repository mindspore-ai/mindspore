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

#include <string>
#include <algorithm>
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kConvInputsNum = 2;
constexpr size_t kConvOutputsNum = 1;
}  // namespace

bool ConvCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                            const std::vector<KernelTensorPtr> &outputs) {
  MS_ERROR_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto prim = base_operator->GetPrim();
  MS_ERROR_IF_NULL(prim);
  format_ = GetValue<std::string>(prim->GetAttr(kAttrFormat));
  group_ = GetValue<int64_t>(prim->GetAttr(kAttrGroup));
  pad_mode_ = GetValue<std::string>(prim->GetAttr(kAttrPadMode));
  return true;
}

int ConvCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs,
                             const std::map<uint32_t, tensor::TensorPtr> &) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  auto src_shape = inputs[kIndex0]->GetDeviceShapeAdaptively();
  auto weight_shape = inputs[kIndex1]->GetDeviceShapeAdaptively();
  auto dst_shape = outputs[kIndex0]->GetDeviceShapeAdaptively();
  auto src_dim = src_shape.size();
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
  auto prim = base_operator->GetPrim();
  MS_ERROR_IF_NULL(prim);
  const auto strides_include_nc = GetValue<std::vector<int64_t>>(prim->GetAttr(stride_attr));
  const auto dilation_include_nc = GetValue<std::vector<int64_t>>(prim->GetAttr(dilation_attr));
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
  GetPadding(base_operator, src_shape, padding_info);

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

bool ConvCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                              const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kConvInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kConvOutputsNum, kernel_name_);
  SetArgumentHandle(DNNL_ARG_SRC, inputs[0]->addr);
  SetArgumentHandle(DNNL_ARG_WEIGHTS, inputs[1]->addr);
  SetArgumentHandle(DNNL_ARG_DST, outputs[0]->addr);
  ExecutePrimitive();
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Conv2D, ConvCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Conv3D, ConvCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
