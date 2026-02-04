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

#include "kernel/ascend/atb/kernel_mod_impl/atb_kernel_plugin.h"

#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/framework_utils.h"
#include "include/op_def.h"
#include "kernel/ascend/atb/kernel_mod_impl/atb_kernel_mod.h"
#include "utils/trace_base.h"

namespace mindspore::kernel {

KernelModPtr AtbKernelPlugin::BuildKernel(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);

  std::string opname = common::AnfAlgo::GetCNodeName(anf_node);
  MS_LOG(DEBUG) << "atb op [" << opname << "]";
  std::shared_ptr<ATBKernelMod> kernel_ptr;
  kernel_ptr = Factory<ATBKernelMod>::Instance().Create(opname);
  if (kernel_ptr == nullptr) {
    MS_LOG(ERROR) << "atb can't find Kernel[" << opname << "]";
    return nullptr;
  }
  std::vector<KernelTensor *> input_kernel_tensors = AnfAlgo::GetOrCreateAllInputKernelTensors(anf_node);
  std::vector<KernelTensor *> output_kernel_tensors = AnfAlgo::GetOrCreateAllOutputKernelTensors(anf_node);

  if (!std::static_pointer_cast<KernelMod>(kernel_ptr)
         ->Init(common::AnfAlgo::GetCNodePrimitive(anf_node), input_kernel_tensors, output_kernel_tensors)) {
    MS_LOG_WITH_NODE(EXCEPTION, anf_node)
      << "#dmsg#Kernel build failed:#dmsg#Initialize atb kernel op[" << anf_node->fullname_with_scope() << "] failed."
      << trace::DumpSourceLines(anf_node);
  }

  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (CheckResizeCondition(cnode)) {
    if (kernel_ptr->Resize(input_kernel_tensors, output_kernel_tensors) == KRET_RESIZE_FAILED) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode) << "#dmsg#Kernel build failed:#dmsg#hostapi kernel op["
                                         << cnode->fullname_with_scope() << "] Resize failed.";
    }
  }
  return kernel_ptr;
}

bool AtbKernelPlugin::IsRegisteredKernel(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  std::string op_name = common::AnfAlgo::GetCNodeName(anf_node);
  return Factory<ATBKernelMod>::Instance().IsRegistered(op_name);
}

MS_KERNEL_PLUGIN_FACTORY_REG(AtbKernelPlugin, AtbKernelPlugin);
}  // namespace mindspore::kernel
