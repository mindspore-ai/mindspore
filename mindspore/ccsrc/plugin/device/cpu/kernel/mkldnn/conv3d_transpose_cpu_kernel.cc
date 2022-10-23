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

#include "plugin/device/cpu/kernel/mkldnn/conv3d_transpose_cpu_kernel.h"

#include <string>
#include <algorithm>
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kConv3DTranspose = "Conv3DTranspose";
constexpr size_t kConv3DTransposeInputsNum = 2;
constexpr size_t kConv3DTransposeOutputsNum = 1;
}  // namespace

void Conv3DTransposeCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  std::vector<int64_t> src_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  std::vector<int64_t> weight_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  std::vector<int64_t> dst_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);

  size_t src_dim = src_shape.size();
  if (src_dim != SHAPE_5D) {
    MS_LOG(EXCEPTION) << "Conv3DTranspose only supports 5D input, but got " << src_dim << "D!";
  }
  const auto format = common::AnfAlgo::GetNodeAttr<std::string>(kernel_node, FORMAT);

  if (src_dim == SHAPE_5D && format != NCDHW) {
    MS_LOG(EXCEPTION) << kernel_name_ << " only supports 5D input with format NCDHW, but got format " << format;
  }
  if (AnfAlgo::IsShapesDynamic({src_shape, weight_shape, dst_shape})) {
    return;
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
  const auto pad_mode = common::AnfAlgo::GetNodeAttr<std::string>(kernel_node, PAD_MODE);
  const auto strides_include_nc = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, STRIDES);
  const auto dilation_include_nc = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, DILATIONS);
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

  const auto forward_desc = CreateDesc<dnnl::convolution_forward::desc>(
    dnnl::prop_kind::forward_training, dnnl::algorithm::convolution_auto, src_desc, weights_desc, dst_desc, strides,
    dilates, padding_l, padding_r);
  const auto forward_prim_desc = CreateDesc<dnnl::convolution_forward::primitive_desc>(forward_desc, engine_);
  const auto backward_desc = CreateDesc<dnnl::convolution_backward_data::desc>(
    dnnl::algorithm::convolution_auto, src_desc, weights_desc, dst_desc, strides, dilates, padding_l, padding_r);
  const auto backward_prim_desc =
    CreateDesc<dnnl::convolution_backward_data::primitive_desc>(backward_desc, engine_, forward_prim_desc);
  primitive_ = CreatePrimitive<dnnl::convolution_backward_data>(backward_prim_desc);
  AddArgument(DNNL_ARG_DIFF_SRC, src_desc);
  AddArgument(DNNL_ARG_WEIGHTS, weights_desc);
  AddArgument(DNNL_ARG_DIFF_DST, dst_desc);
}

bool Conv3DTransposeCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                         const std::vector<kernel::AddressPtr> &,
                                         const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kConv3DTransposeInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kConv3DTransposeOutputsNum, kernel_name_);
  SetArgumentHandle(DNNL_ARG_DIFF_SRC, outputs[0]->addr);
  SetArgumentHandle(DNNL_ARG_WEIGHTS, inputs[1]->addr);
  SetArgumentHandle(DNNL_ARG_DIFF_DST, inputs[0]->addr);
  ExecutePrimitive();
  return true;
}

std::vector<KernelAttr> Conv3DTransposeCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)};
  return support_list;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Conv3DTranspose,
                                 []() { return std::make_shared<Conv3DTransposeCpuKernelMod>(kConv3DTranspose); });
}  // namespace kernel
}  // namespace mindspore
