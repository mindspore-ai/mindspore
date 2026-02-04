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

#include "kernel/ascend/custom/kernel_mod_impl/custom_kernel_plugin.h"

#include <string>
#include <utility>
#include <vector>

#include "kernel/ascend/custom/kernel_mod_impl/custom_kernel_factory.h"
#include "plugin/ascend/kernel_executor/kernel_select_ascend.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/framework_utils.h"
#include "kernel/ascend/custom/kernel_mod_impl/py_func_kernel_mod.h"

namespace mindspore::kernel {
KernelModPtr CustomKernelPlugin::BuildKernel(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);

  std::string op_fullname = anf_node->fullname_with_scope();
  std::string opname = common::AnfAlgo::GetCNodeName(anf_node);
  // Easy to compare accuracy and performance, later changed to debug
  KernelModPtr kernel_ptr;
  if (CustomKernelFactory::Instance().IsRegistered(opname)) {
    MS_LOG(INFO) << "Supported by CustomKernel: " << opname;
    kernel_ptr = std::static_pointer_cast<KernelMod>(CustomKernelFactory::Instance().Create(opname));
  }

  if (kernel_ptr == nullptr) {
    MS_LOG(ERROR) << "CustomKernel can't find Kernel[" << opname << "]";
    return nullptr;
  }
  kernel_ptr->set_fullname(op_fullname);
  std::vector<KernelTensor *> input_kernel_tensors = AnfAlgo::GetOrCreateAllInputKernelTensors(anf_node);
  std::vector<KernelTensor *> output_kernel_tensors = AnfAlgo::GetOrCreateAllOutputKernelTensors(anf_node);
  if (!kernel_ptr->Init(common::AnfAlgo::GetCNodePrimitive(anf_node), input_kernel_tensors, output_kernel_tensors)) {
    MS_LOG_WITH_NODE(EXCEPTION, anf_node) << "#dmsg#Kernel build failed:#dmsg#Initialize internal kernel op["
                                          << anf_node->fullname_with_scope() << "] failed.";
  }
  return kernel_ptr;
}

bool CustomKernelPlugin::IsRegisteredKernel(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  std::string opname = common::AnfAlgo::GetCNodeName(anf_node);
  auto cnode = anf_node->cast<CNodePtr>();
  if (cnode != nullptr && common::AnfAlgo::HasNodeAttr(kAttrPrimPyFunc, cnode)) {
    mindspore::kernel::CustomKernelFactory::Instance().Register(opname,
                                                                []() { return std::make_shared<PyFuncKernelMod>(); });
  }

  return CustomKernelFactory::Instance().IsRegistered(opname);
}

MS_KERNEL_PLUGIN_FACTORY_REG(CustomKernelPlugin, CustomKernelPlugin);
}  // namespace mindspore::kernel
