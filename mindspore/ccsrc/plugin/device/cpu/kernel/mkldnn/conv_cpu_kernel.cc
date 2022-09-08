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

void ConvCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  std::vector<int64_t> src_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  std::vector<int64_t> weight_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  std::vector<int64_t> dst_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);

  size_t src_dim = src_shape.size();
  if (AnfAlgo::IsShapesDynamic({src_shape, weight_shape, dst_shape})) {
    return;
  }
  if (src_dim != SHAPE_4D && src_dim != SHAPE_5D) {
    MS_LOG(EXCEPTION) << "Conv only supports 4D/5D input, but got " << src_dim << "D!";
  }
  const auto format = common::AnfAlgo::GetNodeAttr<std::string>(kernel_node, FORMAT);
  if (src_dim == SHAPE_4D && format != NCHW) {
    MS_LOG(EXCEPTION) << kernel_name_ << " only supports 4D input with format NCHW, but got format " << format;
  }
  if (src_dim == SHAPE_5D && format != NCDHW) {
    MS_LOG(EXCEPTION) << kernel_name_ << " only supports 5D input with format NCDHW, but got format " << format;
  }
  dnnl::memory::dims kernel_size(weight_shape.begin() + NC_LEN, weight_shape.end());
  const size_t group = LongToSize(common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, GROUP));
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
  const auto stride_attr = src_dim == SHAPE_4D ? STRIDE : STRIDES;
  const auto dilation_attr = src_dim == SHAPE_4D ? DILATION : DILATIONS;
  const auto pad_mode = common::AnfAlgo::GetNodeAttr<std::string>(kernel_node, PAD_MODE);
  const auto strides_include_nc = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, stride_attr);
  const auto dilation_include_nc = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, dilation_attr);
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
  PaddingInfo padding_info{pad_mode, kernel_size, strides, dilation, &padding_l, &padding_r};
  GetPadding(kernel_node, src_shape, padding_info);

  const auto desc = CreateDesc<dnnl::convolution_forward::desc>(
    dnnl::prop_kind::forward_training, dnnl::algorithm::convolution_auto, src_desc, weights_desc, dst_desc, strides,
    dilates, padding_l, padding_r);
  const auto prim_desc = CreateDesc<dnnl::convolution_forward::primitive_desc>(desc, engine_);
  primitive_ = CreatePrimitive<dnnl::convolution_forward>(prim_desc);
  AddArgument(DNNL_ARG_SRC, src_desc);
  AddArgument(DNNL_ARG_WEIGHTS, weights_desc);
  AddArgument(DNNL_ARG_DST, dst_desc);
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
