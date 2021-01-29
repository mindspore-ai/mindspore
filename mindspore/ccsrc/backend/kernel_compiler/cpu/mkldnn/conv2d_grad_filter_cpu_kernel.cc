/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "backend/kernel_compiler/cpu/mkldnn/conv2d_grad_filter_cpu_kernel.h"
#include <string>
#include <algorithm>
#include "utils/ms_utils.h"
#include "backend/kernel_compiler/cpu/mkldnn/mkl_kernel_engine.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
void Conv2dGradFilterCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::vector<size_t> src_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  std::vector<size_t> weight_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  std::vector<size_t> dst_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  if (src_shape.size() != 4 || weight_shape.size() != 4) {
    MS_LOG(EXCEPTION) << ("conv2d grad filter only support nchw input!");
  }
  std::vector<size_t> kernel_size({weight_shape[2], weight_shape[3]});
  size_t group = LongToSize(AnfAlgo::GetNodeAttr<int64_t>(kernel_node, GROUP));
  if (group != 1) {
    if (src_shape[1] % group != 0) {
      MS_LOG(EXCEPTION) << "conv2d channels should be divided by group!";
    }
    weight_shape.insert(weight_shape.begin(), group);
    weight_shape[1] = weight_shape[1] / group;
  }
  dnnl::memory::desc src_desc = GetDefaultMemDesc(src_shape);
  dnnl::memory::desc weights_desc = GetDefaultMemDesc(weight_shape);
  dnnl::memory::desc dst_desc = GetDefaultMemDesc(dst_shape);
  std::vector<int> stride_ori;
  std::vector<int> dilation_ori;
  auto stride_me = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, STRIDE);
  auto dilation_me = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, DILATION);
  (void)std::transform(stride_me.begin(), stride_me.end(), std::back_inserter(stride_ori),
                       [](const int64_t &value) { return static_cast<int>(value); });
  (void)std::transform(dilation_me.begin(), dilation_me.end(), std::back_inserter(dilation_ori),
                       [](const int64_t &value) { return static_cast<int>(value); });

  if (dilation_ori.size() != 4) {
    MS_LOG(EXCEPTION) << "Conv2dGradFilterCPUKernel dilation must be 4d!";
  }
  if (dilation_ori[0] != 1 || dilation_ori[1] != 1) {
    MS_LOG(EXCEPTION) << "Conv2dGradFilterCPUKernel dilation only support 1 in N axis and C axis!";
  }
  std::vector<int> stride{stride_ori[0], stride_ori[1]};
  std::vector<int> dilation{dilation_ori[2], dilation_ori[3]};
  dnnl::memory::dims strides{stride_ori[0], stride_ori[1]};
  dnnl::memory::dims dilates{dilation_ori[2] - 1, dilation_ori[3] - 1};
  const std::string pad_mode = AnfAlgo::GetNodeAttr<std::string>(kernel_node, PAD_MODE);
  std::vector<int> int_padding_l;
  std::vector<int> int_padding_r;
  GetPadding(kernel_node, pad_mode, src_shape, kernel_size, stride, &int_padding_l, &int_padding_r, dilation);
  if (int_padding_l.size() != 2 || int_padding_r.size() != 2) {
    MS_LOG(EXCEPTION) << "get padding failed";
  }
  dnnl::memory::dims padding_l{int_padding_l[0], int_padding_l[1]};
  dnnl::memory::dims padding_r{int_padding_r[0], int_padding_r[1]};
  dnnl::convolution_forward::desc forward_desc =
    dnnl::convolution_forward::desc(dnnl::prop_kind::forward_training, dnnl::algorithm::convolution_auto, src_desc,
                                    weights_desc, dst_desc, strides, dilates, padding_l, padding_r);

  auto forward_prim_desc = dnnl::convolution_forward::primitive_desc(forward_desc, MKLKernelEngine::Get().engine());

  dnnl::convolution_backward_weights::desc backward_desc = dnnl::convolution_backward_weights::desc(
    dnnl::algorithm::convolution_auto, src_desc, weights_desc, dst_desc, strides, dilates, padding_l, padding_r);

  auto backward_prim_desc = dnnl::convolution_backward_weights::primitive_desc(
    backward_desc, MKLKernelEngine::Get().engine(), forward_prim_desc);
  primitive_ = std::make_shared<dnnl::convolution_backward_weights>(backward_prim_desc);

  AddArgument(DNNL_ARG_SRC, src_desc);
  AddArgument(DNNL_ARG_DIFF_DST, dst_desc);
  AddArgument(DNNL_ARG_DIFF_WEIGHTS, weights_desc);
}

bool Conv2dGradFilterCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> & /*workspace*/,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.size() < 2 || outputs.empty()) {
    MS_LOG(EXCEPTION) << "error input output size!";
  }
  SetArgumentHandle(DNNL_ARG_SRC, inputs[1]->addr);
  SetArgumentHandle(DNNL_ARG_DIFF_DST, inputs[0]->addr);
  SetArgumentHandle(DNNL_ARG_DIFF_WEIGHTS, outputs[0]->addr);
  ExecutePrimitive();
  return true;
}
}  // namespace kernel
}  // namespace mindspore
