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
#include "device/cpu/kernel/mkldnn/relu_grad_cpu_kernel.h"
#include "device/cpu/kernel/mkldnn/mkl_kernel_engine.h"
#include "device/cpu/cpu_device_address.h"
#include "common/utils.h"

namespace mindspore {
namespace device {
namespace cpu {
void ReluGradCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::vector<size_t> src_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  if (src_shape.size() != 4 && src_shape.size() != 2) {
    MS_LOG(EXCEPTION) << "relu grad kernel dims invalid " << src_shape.size();
  }
  dnnl::memory::desc src_desc = GetDefaultMemDesc(src_shape);

  dnnl::eltwise_forward::desc forward_desc =
    dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_training, dnnl::algorithm::eltwise_relu, src_desc, 0.0);
  auto forward_prim_desc = dnnl::eltwise_forward::primitive_desc(forward_desc, MKLKernelEngine::Get().engine());

  dnnl::eltwise_backward::desc backward_desc =
    dnnl::eltwise_backward::desc(dnnl::algorithm::eltwise_relu, src_desc, src_desc, 0.0, 0.0);
  auto backward_prim_desc =
    dnnl::eltwise_backward::primitive_desc(backward_desc, MKLKernelEngine::Get().engine(), forward_prim_desc);
  primitive_ = std::make_shared<dnnl::eltwise_backward>(backward_prim_desc);

  AddArgument(DNNL_ARG_SRC, src_desc);
  AddArgument(DNNL_ARG_DIFF_SRC, src_desc);
  AddArgument(DNNL_ARG_DIFF_DST, src_desc);
}

bool ReluGradCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                               const std::vector<kernel::AddressPtr> & /*workspace*/,
                               const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.size() < 2 || outputs.empty()) {
    MS_LOG(EXCEPTION) << "relu grad error input output size!";
  }
  if (inputs[0]->size != outputs[0]->size) {
    MS_LOG(EXCEPTION) << "relu grad error input output data size!";
  }

  SetArgumentHandle(DNNL_ARG_SRC, inputs[1]->addr);
  SetArgumentHandle(DNNL_ARG_DIFF_SRC, inputs[0]->addr);
  SetArgumentHandle(DNNL_ARG_DIFF_DST, inputs[0]->addr);
  ExecutePrimitive();
  size_t mem_bits = outputs[0]->size;
  auto ret = memcpy_s(outputs[0]->addr, mem_bits, inputs[0]->addr, mem_bits);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno " << ret;
    return false;
  }
  return true;
}
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
