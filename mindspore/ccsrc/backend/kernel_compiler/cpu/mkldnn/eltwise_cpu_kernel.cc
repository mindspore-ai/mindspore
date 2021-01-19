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
#include "backend/kernel_compiler/cpu/mkldnn/eltwise_cpu_kernel.h"

#include <string>
#include "backend/kernel_compiler/cpu/mkldnn/mkl_kernel_engine.h"
#include "runtime/device/cpu/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
dnnl::eltwise_forward::desc EltWiseCPUKernel::GetForwardEltwiseDesc(const CNodePtr &kernel_node,
                                                                    dnnl::memory::desc src_desc) {
  std::string kernel_name = AnfAlgo::GetCNodeName(kernel_node);
  if (kernel_name == "ReLU") {
    return dnnl::eltwise_forward::desc(DnnlForward, dnnl::algorithm::eltwise_relu, src_desc, 0.0);
  } else if (kernel_name == "ReLU6") {
    return dnnl::eltwise_forward::desc(DnnlForward, dnnl::algorithm::eltwise_clip, src_desc, 0.0, 6.0);
  } else if (kernel_name == "Abs") {
    return dnnl::eltwise_forward::desc(DnnlForward, dnnl::algorithm::eltwise_abs, src_desc);
  } else if (kernel_name == "Exp") {
    return dnnl::eltwise_forward::desc(DnnlForward, dnnl::algorithm::eltwise_exp, src_desc);
  } else if (kernel_name == "Log") {
    return dnnl::eltwise_forward::desc(DnnlForward, dnnl::algorithm::eltwise_log, src_desc);
  } else if (kernel_name == "Sigmoid") {
    return dnnl::eltwise_forward::desc(DnnlForward, dnnl::algorithm::eltwise_logistic, src_desc);
  } else if (kernel_name == "Sqrt") {
    return dnnl::eltwise_forward::desc(DnnlForward, dnnl::algorithm::eltwise_sqrt, src_desc);
  } else if (kernel_name == "Square") {
    return dnnl::eltwise_forward::desc(DnnlForward, dnnl::algorithm::eltwise_square, src_desc);
  } else if (kernel_name == "Tanh") {
    return dnnl::eltwise_forward::desc(DnnlForward, dnnl::algorithm::eltwise_tanh, src_desc);
  } else if (kernel_name == "Elu") {
    return dnnl::eltwise_forward::desc(DnnlForward, dnnl::algorithm::eltwise_elu, src_desc, 1.0);
  } else {
    MS_LOG(EXCEPTION) << "Eltwise operators don't support " << kernel_name;
  }
}

void EltWiseCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::vector<size_t> src_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  if (src_shape.size() == 0) {
    src_shape.insert(src_shape.begin(), 1);
  }
  dnnl::memory::desc src_desc = GetDefaultMemDesc(src_shape);

  auto desc = GetForwardEltwiseDesc(kernel_node, src_desc);
  auto prim_desc = dnnl::eltwise_forward::primitive_desc(desc, MKLKernelEngine::Get().engine());
  primitive_ = std::make_shared<dnnl::eltwise_forward>(prim_desc);

  AddArgument(DNNL_ARG_SRC, src_desc);
  AddArgument(DNNL_ARG_DST, src_desc);
}

bool EltWiseCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
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
