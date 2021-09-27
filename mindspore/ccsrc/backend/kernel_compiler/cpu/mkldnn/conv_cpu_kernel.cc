/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/mkldnn/conv_cpu_kernel.h"
#include <string>
#include <algorithm>
#include "utils/ms_utils.h"
#include "backend/kernel_compiler/cpu/mkldnn/mkl_kernel_engine.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kConvInputsNum = 2;
constexpr size_t kConvOutputsNum = 1;
constexpr size_t kShapeSize4D = 4;
constexpr size_t kShapeSize5D = 5;
constexpr size_t kKernelStartAxis = 2;
}  // namespace

void ConvCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  std::vector<size_t> src_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  std::vector<size_t> weight_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  std::vector<size_t> dst_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  size_t src_dim = src_shape.size();
  size_t weight_dim = weight_shape.size();
  if (src_dim < kShapeSize4D || src_dim > kShapeSize5D || src_dim != weight_dim) {
    MS_LOG(EXCEPTION) << "Conv only supports 4D/5D input!";
  }
  std::vector<size_t> kernel_size;
  for (size_t i = kKernelStartAxis; i < src_dim; ++i) {
    (void)kernel_size.emplace_back(weight_shape[i]);
  }
  size_t group = LongToSize(AnfAlgo::GetNodeAttr<int64_t>(kernel_node, GROUP));
  if (group > 1) {
    if (src_shape[1] % group != 0) {
      MS_LOG(EXCEPTION) << "Conv channels should be divided by group!";
    }
    (void)weight_shape.insert(weight_shape.begin(), group);
    weight_shape[1] = weight_shape[1] / group;
  }
  dnnl::memory::desc src_desc = GetDefaultMemDesc(src_shape);
  dnnl::memory::desc weights_desc = GetDefaultMemDesc(weight_shape);
  dnnl::memory::desc dst_desc = GetDefaultMemDesc(dst_shape);
  std::vector<int> stride_ori;
  std::vector<int> dilation_ori;
  auto stride_attr = src_dim == kShapeSize4D ? STRIDE : STRIDES;
  auto dilation_attr = src_dim == kShapeSize4D ? DILATION : DILATIONS;
  auto stride_me = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, stride_attr);
  auto dilation_me = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, dilation_attr);
  (void)std::transform(stride_me.begin(), stride_me.end(), std::back_inserter(stride_ori),
                       [](const int64_t &value) { return LongToInt(value); });
  (void)std::transform(dilation_me.begin(), dilation_me.end(), std::back_inserter(dilation_ori),
                       [](const int64_t &value) { return LongToInt(value); });
  if (stride_ori.size() != src_dim) {
    MS_LOG(EXCEPTION) << "Conv stride size must be " << src_dim << "D!";
  }
  if (stride_ori[0] != 1 || stride_ori[1] != 1) {
    MS_LOG(EXCEPTION) << "Conv2d stride only support 1 in N axis and C axis!";
  }
  if (dilation_ori.size() != src_dim) {
    MS_LOG(EXCEPTION) << "Conv dilation size must be " << src_dim << "D!";
  }
  if (dilation_ori[0] != 1 || dilation_ori[1] != 1) {
    MS_LOG(EXCEPTION) << "Conv2d dilation only support 1 in N axis and C axis!";
  }

  std::vector<int> stride;
  std::vector<int> dilation;
  dnnl::memory::dims strides;
  dnnl::memory::dims dilates;
  for (size_t i = kKernelStartAxis; i < src_dim; ++i) {
    (void)stride.emplace_back(stride_ori[i]);
    (void)strides.emplace_back(stride_ori[i]);
    (void)dilation.emplace_back(dilation_ori[i]);
    (void)dilates.emplace_back(dilation_ori[i] - 1);
  }
  std::vector<int> int_padding_l;
  std::vector<int> int_padding_r;
  const std::string pad_mode = AnfAlgo::GetNodeAttr<std::string>(kernel_node, PAD_MODE);
  GetPadding(kernel_node, pad_mode, src_shape, kernel_size, stride, &int_padding_l, &int_padding_r, dilation);
  if (int_padding_l.size() + kKernelStartAxis != src_dim || int_padding_r.size() + kKernelStartAxis != src_dim) {
    MS_LOG(EXCEPTION) << "Get padding failed!";
  }
  dnnl::memory::dims padding_l;
  dnnl::memory::dims padding_r;
  for (size_t i = 0; i < int_padding_l.size(); ++i) {
    (void)padding_l.emplace_back(int_padding_l[i]);
    (void)padding_r.emplace_back(int_padding_r[i]);
  }
  dnnl::convolution_forward::desc desc =
    dnnl::convolution_forward::desc(dnnl::prop_kind::forward_training, dnnl::algorithm::convolution_auto, src_desc,
                                    weights_desc, dst_desc, strides, dilates, padding_l, padding_r);

  auto prim_desc = dnnl::convolution_forward::primitive_desc(desc, MKLKernelEngine::Get().engine());
  primitive_ = std::make_shared<dnnl::convolution_forward>(prim_desc);
  AddArgument(DNNL_ARG_SRC, src_desc);
  AddArgument(DNNL_ARG_WEIGHTS, weights_desc);
  AddArgument(DNNL_ARG_DST, dst_desc);
}

bool ConvCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                           const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kConvInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kConvOutputsNum, kernel_name_);
  SetArgumentHandle(DNNL_ARG_SRC, inputs[0]->addr);
  SetArgumentHandle(DNNL_ARG_WEIGHTS, inputs[1]->addr);
  SetArgumentHandle(DNNL_ARG_DST, outputs[0]->addr);
  ExecutePrimitive();
  return true;
}
}  // namespace kernel
}  // namespace mindspore
