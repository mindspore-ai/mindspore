/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/mkldnn/addn_cpu_kernel.h"
#include "backend/kernel_compiler/cpu/mkldnn/mkl_kernel_engine.h"
#include "runtime/device/cpu/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
void AddNCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  input_num_ = AnfAlgo::GetInputTensorNum(kernel_node);
  CheckParam(kernel_node);
  std::vector<size_t> src0_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  std::vector<size_t> src1_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  std::vector<size_t> dst_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  dnnl::memory::desc src0_mem_desc = GetDefaultMemDesc(src0_shape);
  dnnl::memory::desc src1_mem_desc = GetDefaultMemDesc(src1_shape);
  dnnl::memory::desc dst_mem_desc = GetDefaultMemDesc(dst_shape);
  dnnl::binary::desc desc = dnnl::binary::desc(dnnl::algorithm::binary_add, src0_mem_desc, src1_mem_desc, dst_mem_desc);
  auto prim_desc = dnnl::binary::primitive_desc(desc, MKLKernelEngine::Get().engine());
  primitive_ = std::make_shared<dnnl::binary>(prim_desc);
  AddArgument(DNNL_ARG_SRC_0, src0_mem_desc);
  AddArgument(DNNL_ARG_SRC_1, src1_mem_desc);
  AddArgument(DNNL_ARG_DST, dst_mem_desc);
}

bool AddNCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                           const std::vector<kernel::AddressPtr> & /*workspace*/,
                           const std::vector<kernel::AddressPtr> &outputs) {
  SetArgumentHandle(DNNL_ARG_SRC_0, inputs[0]->addr);
  SetArgumentHandle(DNNL_ARG_SRC_1, inputs[1]->addr);
  SetArgumentHandle(DNNL_ARG_DST, outputs[0]->addr);
  ExecutePrimitive();
  for (size_t index = 2; index < input_num_; ++index) {
    SetArgumentHandle(DNNL_ARG_SRC_0, outputs[0]->addr);
    SetArgumentHandle(DNNL_ARG_SRC_1, inputs[index]->addr);
    SetArgumentHandle(DNNL_ARG_DST, outputs[0]->addr);
    ExecutePrimitive();
  }
  return true;
}

void AddNCPUKernel::CheckParam(const CNodePtr &kernel_node) {
  auto src0_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  auto dst_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  if (src0_shape != dst_shape) {
    MS_LOG(EXCEPTION) << "AddN output shape must be equal to input shape.";
  }
  for (size_t index = 1; index < input_num_; ++index) {
    auto src_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, index);
    if (src0_shape != src_shape) {
      MS_LOG(EXCEPTION) << "AddN input shapes must be equal.";
    }
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << "Output number is " << output_num << ", but AddNCPUKernel needs 1 output.";
  }
}
}  // namespace kernel
}  // namespace mindspore
