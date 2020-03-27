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
#include "device/cpu/kernel/mkldnn/softmax_cpu_kernel.h"
#include "device/cpu/kernel/mkldnn/mkl_kernel_engine.h"
#include "device/cpu/cpu_device_address.h"
#include "common/utils.h"

namespace mindspore {
namespace device {
namespace cpu {
void SoftmaxCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::vector<size_t> src_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  std::vector<int> axis_list = AnfAlgo::GetNodeAttr<std::vector<int>>(kernel_node, AXIS);
  if (axis_list.size() != 1) {
    MS_LOG(EXCEPTION) << "cpu softmax only support input axis size 1";
  }
  int axis = axis_list[0];
  if (axis == -1 || axis >= SizeToInt(src_shape.size())) {
    axis = SizeToInt(src_shape.size()) - 1;
  }
  dnnl::memory::desc src_desc = GetDefaultMemDesc(src_shape);
  dnnl::softmax_forward::desc desc = dnnl::softmax_forward::desc(dnnl::prop_kind::forward_training, src_desc, axis);
  auto prim_desc = dnnl::softmax_forward::primitive_desc(desc, MKLKernelEngine::Get().engine());
  primitive_ = std::make_shared<dnnl::softmax_forward>(prim_desc);
  AddArgument(DNNL_ARG_SRC, src_desc);
  AddArgument(DNNL_ARG_DST, src_desc);
}

bool SoftmaxCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                              const std::vector<kernel::AddressPtr> & /*workspace*/,
                              const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(EXCEPTION) << "softmax error input output size!";
  }
  SetArgumentHandle(DNNL_ARG_SRC, inputs[0]->addr);
  SetArgumentHandle(DNNL_ARG_DST, outputs[0]->addr);
  ExecutePrimitive();
  return true;
}
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
