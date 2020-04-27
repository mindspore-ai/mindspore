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
#include "kernel/cpu/mkldnn/conv2d_cpu_kernel.h"
#include <string>
#include "common/utils.h"
#include "kernel/cpu/mkldnn/mkl_kernel_engine.h"
#include "device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
void Conv2dCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::vector<size_t> src_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  std::vector<size_t> weight_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  std::vector<size_t> dst_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  if (src_shape.size() != 4 || weight_shape.size() != 4) {
    MS_LOG(EXCEPTION) << "conv2d only support nchw input!";
  }
  dnnl::memory::desc src_desc = GetDefaultMemDesc(src_shape);
  dnnl::memory::desc weights_desc = GetDefaultMemDesc(weight_shape);
  dnnl::memory::desc dst_desc = GetDefaultMemDesc(dst_shape);

  int kernel_size = SizeToInt(weight_shape[3]);
  auto stride_ori = AnfAlgo::GetNodeAttr<std::vector<int>>(kernel_node, STRIDE);
  auto dilation_ori = AnfAlgo::GetNodeAttr<std::vector<int>>(kernel_node, DILATION);
  if (stride_ori.size() != 4 || stride_ori[2] != stride_ori[3]) {
    MS_LOG(EXCEPTION) << "conv2d only support equal stride, and stride must be 4d!";
  }
  if (stride_ori[0] != 1 || stride_ori[1] != 1) {
    MS_LOG(EXCEPTION) << "conv2d stride only support 1 in N axis and C axis!";
  }
  if (dilation_ori.size() != 4 || dilation_ori[2] != 1 || dilation_ori[3] != 1) {
    MS_LOG(EXCEPTION) << "conv2d dilation only support 1, and dilation must be 4d!";
  }
  if (dilation_ori[0] != 1 || dilation_ori[1] != 1) {
    MS_LOG(EXCEPTION) << "conv2d dilation only support 1 in N axis and C axis!";
  }
  int stride = stride_ori[2];
  int dilation = dilation_ori[2];

  dnnl::memory::dims strides{stride, stride};
  dnnl::memory::dims dilates{dilation - 1, dilation - 1};
  std::vector<int> int_padding_l;
  std::vector<int> int_padding_r;

  const std::string pad_mode = AnfAlgo::GetNodeAttr<std::string>(kernel_node, PAD_MODE);
  GetPadding(kernel_node, pad_mode, src_shape, kernel_size, stride, &int_padding_l, &int_padding_r);
  if (int_padding_l.size() != 2 || int_padding_r.size() != 2) {
    MS_LOG(EXCEPTION) << "get padding failed";
  }
  dnnl::memory::dims padding_l{int_padding_l[0], int_padding_l[1]};
  dnnl::memory::dims padding_r{int_padding_r[0], int_padding_r[1]};
  dnnl::convolution_forward::desc desc =
    dnnl::convolution_forward::desc(dnnl::prop_kind::forward_training, dnnl::algorithm::convolution_auto, src_desc,
                                    weights_desc, dst_desc, strides, dilates, padding_l, padding_r);

  auto prim_desc = dnnl::convolution_forward::primitive_desc(desc, MKLKernelEngine::Get().engine());
  primitive_ = std::make_shared<dnnl::convolution_forward>(prim_desc);

  AddArgument(DNNL_ARG_SRC, src_desc);
  AddArgument(DNNL_ARG_WEIGHTS, weights_desc);
  AddArgument(DNNL_ARG_DST, dst_desc);
}

bool Conv2dCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                             const std::vector<kernel::AddressPtr> & /*workspace*/,
                             const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.size() < 2 || outputs.empty()) {
    MS_LOG(EXCEPTION) << "error input output size!";
  }
  SetArgumentHandle(DNNL_ARG_SRC, inputs[0]->addr);
  SetArgumentHandle(DNNL_ARG_WEIGHTS, inputs[1]->addr);
  SetArgumentHandle(DNNL_ARG_DST, outputs[0]->addr);
  ExecutePrimitive();
  return true;
}
}  // namespace kernel
}  // namespace mindspore
