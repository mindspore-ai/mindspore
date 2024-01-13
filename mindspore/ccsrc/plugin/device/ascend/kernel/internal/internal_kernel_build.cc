/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include <string>
#include <utility>

#include "plugin/device/ascend/kernel/internal/internal_kernel_build.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_mod.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/factory/ms_factory.h"
#include "kernel/framework_utils.h"

namespace mindspore {
namespace kernel {
KernelModPtr InternalKernelBuild(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);

  std::string opname = common::AnfAlgo::GetCNodeName(anf_node);
  MS_LOG(DEBUG) << "internal op [" << opname << "]";
  auto kernel_ptr = Factory<InternalKernelMod>::Instance().Create(opname);
  if (kernel_ptr == nullptr) {
    MS_LOG(ERROR) << "internal can't find Kernel[" << opname << "]";
    return nullptr;
  }
  std::vector<KernelTensor *> input_kernel_tensors = AnfAlgo::GetOrCreateAllInputKernelTensors(anf_node);
  std::vector<KernelTensor *> output_kernel_tensors = AnfAlgo::GetOrCreateAllOutputKernelTensors(anf_node);

  if (!std::static_pointer_cast<KernelMod>(kernel_ptr)
         ->Init(common::AnfAlgo::GetCNodePrimitive(anf_node), input_kernel_tensors, output_kernel_tensors)) {
    MS_LOG(EXCEPTION) << "#dmsg#Kernel build failed:#dmsg#Initialize aclnn kernel op["
                      << anf_node->fullname_with_scope() << "] failed.";
  }

  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (CheckResizeCondition(cnode)) {
    if (kernel_ptr->Resize(input_kernel_tensors, output_kernel_tensors) == KRET_RESIZE_FAILED) {
      MS_LOG(EXCEPTION) << "#dmsg#Kernel build failed:#dmsg#hostapi kernel op[" << cnode->fullname_with_scope()
                        << "] Resize failed.";
    }
  }

  return kernel_ptr;
}

bool IsRegisteredInternalKernel(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  std::string opname = common::AnfAlgo::GetCNodeName(anf_node);
  return Factory<InternalKernelMod>::Instance().IsRegistered(opname);
}
}  // namespace kernel
}  // namespace mindspore
