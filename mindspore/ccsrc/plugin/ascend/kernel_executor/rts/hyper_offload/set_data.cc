/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "plugin/ascend/kernel_executor/rts/hyper_offload/set_data.h"
#include "include/utils/anfalgo.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/framework_utils.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"

namespace mindspore {
namespace kernel {
bool SetDataKernel::Init(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  std::vector<KernelTensor *> input_kernel_tensors = AnfAlgo::GetOrCreateAllInputKernelTensors(anf_node);
  std::vector<KernelTensor *> output_kernel_tensors = AnfAlgo::GetOrCreateAllOutputKernelTensors(anf_node);

  auto prim = common::AnfAlgo::GetCNodePrimitive(anf_node);
  MS_EXCEPTION_IF_NULL(prim);
  primitive_ = prim;
  kernel_name_ = prim->name();
  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (CheckResizeCondition(cnode)) {
    if (Resize(input_kernel_tensors, output_kernel_tensors) == KRET_RESIZE_FAILED) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode) << "#dmsg#Kernel build failed:#dmsg#rts kernel op["
                                         << cnode->fullname_with_scope() << "] Resize failed.";
    }
  }
  return true;
}

bool SetDataKernel::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                           const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_LOG(DEBUG) << "Begin to launch kernel: " << kernel_name_;
  auto input = inputs[0];
  MS_EXCEPTION_IF_NULL(input);
  auto value = inputs[1];
  MS_EXCEPTION_IF_NULL(value);
  auto input_device_tensor = input->device_address();
  MS_EXCEPTION_IF_NULL(input_device_tensor);
  auto value_device_tensor = value->device_address();
  MS_EXCEPTION_IF_NULL(value_device_tensor);
  input_device_tensor->set_device_pointer(value_device_tensor->device_pointer());
  input->SetDeviceType(value->GetDeviceType());

  // Reset input and value ref count.
  input->IncreaseNewRefCount(value->new_ref_count());
  value->set_new_ref_count(input->new_ref_count());

  // Update output storage.
  auto output = outputs[0];
  MS_EXCEPTION_IF_NULL(output);
  auto output_device_tensor = output->device_address();
  MS_EXCEPTION_IF_NULL(output_device_tensor);
  output_device_tensor->set_device_pointer(value_device_tensor->device_pointer());
  output->SetDeviceType(value->GetDeviceType());
  if (value->tensor_storage_info() != nullptr) {
    input->set_tensor_storage_info(value->tensor_storage_info());
    output->set_tensor_storage_info(value->tensor_storage_info());
  }
  MS_LOG(DEBUG) << "End to launch kernel: " << kernel_name_;
  return true;
}
}  // namespace kernel
}  // namespace mindspore
