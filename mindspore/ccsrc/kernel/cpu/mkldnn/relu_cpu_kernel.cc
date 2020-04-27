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
#include "kernel/cpu/mkldnn/relu_cpu_kernel.h"
#include "kernel/cpu/mkldnn/mkl_kernel_engine.h"
#include "device/cpu/cpu_device_address.h"
#include "common/utils.h"

namespace mindspore {
namespace kernel {
void ReluCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::vector<size_t> src_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  if (src_shape.size() != 4 && src_shape.size() != 2) {
    MS_LOG(EXCEPTION) << "relu kernel dims invalid " << src_shape.size();
  }
  dnnl::memory::desc src_desc = GetDefaultMemDesc(src_shape);

  dnnl::eltwise_forward::desc desc =
    dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_training, dnnl::algorithm::eltwise_relu, src_desc, 0.0);
  auto prim_desc = dnnl::eltwise_forward::primitive_desc(desc, MKLKernelEngine::Get().engine());
  primitive_ = std::make_shared<dnnl::eltwise_forward>(prim_desc);

  AddArgument(DNNL_ARG_SRC, src_desc);
  AddArgument(DNNL_ARG_DST, src_desc);
}

bool ReluCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                           const std::vector<kernel::AddressPtr> & /*workspace*/,
                           const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(EXCEPTION) << "error input output size!";
  }
  SetArgumentHandle(DNNL_ARG_SRC, inputs[0]->addr);
  SetArgumentHandle(DNNL_ARG_DST, outputs[0]->addr);
  ExecutePrimitive();
  return true;
}
}  // namespace kernel
}  // namespace mindspore
