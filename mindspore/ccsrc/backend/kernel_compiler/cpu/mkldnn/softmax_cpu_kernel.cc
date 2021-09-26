/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/mkldnn/softmax_cpu_kernel.h"
#include <algorithm>
#include "backend/kernel_compiler/cpu/mkldnn/mkl_kernel_engine.h"
#include "runtime/device/cpu/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSoftmaxInputsNum = 1;
constexpr size_t kSoftmaxOutputsNum = 1;
}  // namespace

void SoftmaxCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  std::vector<size_t> src_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  std::vector<int> axis_list;
  std::vector<int64_t> axis_list_me = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, AXIS);
  (void)std::transform(axis_list_me.begin(), axis_list_me.end(), std::back_inserter(axis_list),
                       [](const int64_t &value) { return static_cast<int>(value); });
  if (axis_list.size() != 1) {
    MS_LOG(EXCEPTION) << "Cpu softmax only support input axis size 1!";
  }
  int axis = axis_list[0];
  if (axis >= SizeToInt(src_shape.size())) {
    axis = SizeToInt(src_shape.size()) - 1;
  }
  while (axis < 0) {
    axis += SizeToInt(src_shape.size());
  }
  dnnl::memory::desc src_desc = GetDefaultMemDesc(src_shape);
  dnnl::softmax_forward::desc desc = dnnl::softmax_forward::desc(dnnl::prop_kind::forward_training, src_desc, axis);
  auto prim_desc = dnnl::softmax_forward::primitive_desc(desc, MKLKernelEngine::Get().engine());
  primitive_ = std::make_shared<dnnl::softmax_forward>(prim_desc);
  AddArgument(DNNL_ARG_SRC, src_desc);
  AddArgument(DNNL_ARG_DST, src_desc);
}

bool SoftmaxCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                              const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSoftmaxInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSoftmaxOutputsNum, kernel_name_);
  SetArgumentHandle(DNNL_ARG_SRC, inputs[0]->addr);
  SetArgumentHandle(DNNL_ARG_DST, outputs[0]->addr);
  ExecutePrimitive();
  return true;
}
}  // namespace kernel
}  // namespace mindspore
