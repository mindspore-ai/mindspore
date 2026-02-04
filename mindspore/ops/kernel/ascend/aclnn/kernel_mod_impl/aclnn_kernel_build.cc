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
#include <vector>
#include "kernel/ascend/aclnn/kernel_mod_impl/aclnn_kernel_build.h"
#include "kernel/ascend/aclnn/kernel_mod_impl/aclnn_kernel_mod.h"
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/custom_aclnn_utils.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/framework_utils.h"
#include "include/op_def.h"
#include "utils/trace_base.h"
#include "primitive/framework_ops.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"

namespace mindspore {
namespace kernel {
KernelModPtr AclnnOpBuild(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);

  std::string opname = common::AnfAlgo::GetCNodeName(anf_node);
  MS_LOG(DEBUG) << "aclnn op [" << opname << "]";
  std::shared_ptr<AclnnKernelMod> kernel_ptr;
  kernel_ptr = IsPrimitiveCNode(anf_node, prim::kPrimCustom) ? GetCustomAclnnKernelMod(anf_node)
                                                             : Factory<AclnnKernelMod>::Instance().Create(opname);
  if (kernel_ptr == nullptr) {
    MS_LOG(ERROR) << "aclnn can't find Kernel[" << opname << "]";
    return nullptr;
  }
  std::vector<KernelTensor *> input_kernel_tensors = AnfAlgo::GetOrCreateAllInputKernelTensors(anf_node);
  std::vector<KernelTensor *> output_kernel_tensors = AnfAlgo::GetOrCreateAllOutputKernelTensors(anf_node);

  if (!std::static_pointer_cast<KernelMod>(kernel_ptr)
         ->Init(common::AnfAlgo::GetCNodePrimitive(anf_node), input_kernel_tensors, output_kernel_tensors)) {
    MS_LOG_WITH_NODE(EXCEPTION, anf_node)
      << "#dmsg#Kernel build failed:#dmsg#Initialize aclnn kernel op[" << anf_node->fullname_with_scope() << "] failed."
      << trace::DumpSourceLines(anf_node);
  }

  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (!CheckResizeCondition(cnode)) {
    static std::once_flag set_dynamic_flag;
    std::call_once(set_dynamic_flag, [kernel_ptr]() { kernel_ptr->SetDynamic(true); });
  }
  device::ascend::AclnnInit();
  return kernel_ptr;
}

bool IsRegisteredAclnnOp(const std::string &op_name) {
  return Factory<AclnnKernelMod>::Instance().IsRegistered(op_name);
}

bool IsEnabledAclnnDispatch(const std::string &op_name) {
  mindspore::ops::OpDefPtr op_def = mindspore::ops::GetOpDef(op_name);
  if (op_def == nullptr) {
    MS_LOG(INFO) << op_name << " is not defined in opdef.";
    return false;
  }
  return op_def->enable_dispatch_;
}

bool IsViewOp(const std::string &op_name) {
  mindspore::ops::OpDefPtr op_def = mindspore::ops::GetOpDef(op_name);
  if (op_def == nullptr) {
    MS_LOG(INFO) << op_name << " is not defined in opdef.";
    return false;
  }
  return op_def->is_view_;
}

bool IsAclnnViewOp(const std::string &op_name) {
  const auto &op_def = mindspore::ops::GetOpDef(op_name);
  const auto &graph_view_prim = op_def != nullptr ? op_def->is_graph_view_ : false;
  return graph_view_prim;
}
}  // namespace kernel
}  // namespace mindspore
