/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/rts/reshape_ext.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "kernel/framework_utils.h"

namespace mindspore {
namespace kernel {
ReshapeExtKernel::~ReshapeExtKernel() {}

bool ReshapeExtKernel::Init(const AnfNodePtr &anf_node) {
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

std::vector<size_t> ReshapeExtKernel::GetLaunchIgnoredInputAddressIdx() const { return {kIndex1}; }

bool ReshapeExtKernel::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                              const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(inputs[kIndex0]);
  MS_EXCEPTION_IF_NULL(outputs[kIndex0]);
  MS_EXCEPTION_IF_NULL(stream_ptr);

  auto status =
    aclrtMemcpyAsync(outputs[kIndex0]->device_ptr(), outputs[kIndex0]->size(), inputs[kIndex0]->device_ptr(),
                     inputs[kIndex0]->size(), ACL_MEMCPY_DEVICE_TO_DEVICE, stream_ptr);
  if (status != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Launch failed. kernel: " << kernel_name_ << ", call aclrtMemcpyAsync failed, ret = 0x" << status;
    return false;
  }

  return true;
}
}  // namespace kernel
}  // namespace mindspore
