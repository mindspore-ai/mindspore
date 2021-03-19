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
#include "backend/kernel_compiler/cpu/mkldnn/log_softmax_cpu_kernel.h"
#include <algorithm>
#include "backend/kernel_compiler/cpu/mkldnn/mkl_kernel_engine.h"
#include "runtime/device/cpu/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
void LogSoftmaxCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::vector<size_t> src_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  int axis = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, AXIS);
  if (axis >= SizeToInt(src_shape.size())) {
    axis = SizeToInt(src_shape.size()) - 1;
  }
  while (axis < 0) {
    axis += SizeToInt(src_shape.size());
  }
  dnnl::memory::desc src_desc = GetDefaultMemDesc(src_shape);
  dnnl::logsoftmax_forward::desc desc =
    dnnl::logsoftmax_forward::desc(dnnl::prop_kind::forward_training, src_desc, axis);
  auto prim_desc = dnnl::logsoftmax_forward::primitive_desc(desc, MKLKernelEngine::Get().engine());
  primitive_ = std::make_shared<dnnl::logsoftmax_forward>(prim_desc);
  AddArgument(DNNL_ARG_SRC, src_desc);
  AddArgument(DNNL_ARG_DST, src_desc);
}

bool LogSoftmaxCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                 const std::vector<kernel::AddressPtr> & /*workspace*/,
                                 const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(EXCEPTION) << "log softmax error input output size!";
  }
  SetArgumentHandle(DNNL_ARG_SRC, inputs[0]->addr);
  SetArgumentHandle(DNNL_ARG_DST, outputs[0]->addr);
  ExecutePrimitive();
  return true;
}
}  // namespace kernel
}  // namespace mindspore
