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

#include "plugin/device/ascend/kernel/hccl/hccl_kernel_build.h"

#include <string>
#include <memory>
#include "plugin/device/ascend/kernel/hccl/hccl_kernel.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "kernel/framework_utils.h"

namespace mindspore {
namespace kernel {
KernelModPtr HcclOpBuild(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto prim = common::AnfAlgo::GetCNodePrimitive(anf_node);
  MS_LOG(INFO) << "Build hccl op [" << prim->name() << "]";

  auto kernel_mod_ptr = HcclKernelFactory::Get(prim->name());
  if (kernel_mod_ptr == nullptr) {
    MS_LOG(ERROR) << "Hccl can't find kernel[" << prim->name() << "]";
    return nullptr;
  }

  auto func_graph = anf_node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  kernel_mod_ptr->SetIsGraphMode(kernel_graph->is_graph_run_mode());

  std::vector<KernelTensor *> input_kernel_tensors = AnfAlgo::GetOrCreateAllInputKernelTensors(anf_node);
  std::vector<KernelTensor *> output_kernel_tensors = AnfAlgo::GetOrCreateAllOutputKernelTensors(anf_node);
  if (!std::static_pointer_cast<KernelMod>(kernel_mod_ptr)->Init(prim, input_kernel_tensors, output_kernel_tensors)) {
    MS_LOG(EXCEPTION) << "#dmsg#Kernel build failed:#dmsg#Initialize hccl kernel op[" << anf_node->fullname_with_scope()
                      << "] failed.";
  }

  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (kernel::CheckResizeCondition(cnode)) {
    kernel_mod_ptr->Resize(input_kernel_tensors, output_kernel_tensors);
  }

  return kernel_mod_ptr;
}
}  // namespace kernel
}  // namespace mindspore
