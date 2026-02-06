/**
 * Copyright 2021-2025 Huawei Technologies Co., Ltd
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
#include "backend/common/pass/custom_op_const_input_to_attr.h"
#include <vector>
#include <string>
#include "primitive/framework_ops.h"
#include "backend/common/pass/const_input_to_attr.h"
#include "include/utils/anfalgo.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/backend/common/pass_manager/helper.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"

namespace mindspore {
namespace opt {
namespace {
constexpr auto kAOTFuncType = "aot";
}
std::vector<std::string> CustomOpConstInputToAttr::MustExistPrimitiveName() const {
  std::vector<std::string> ret{prim::kPrimCustom->name()};
  return ret;
}

const AnfNodePtr CustomOpConstInputToAttr::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                   const EquivPtr &) const {
  if (node == nullptr || !AnfUtils::IsRealCNodeKernel(node)) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (!IsPrimitiveCNode(cnode, prim::kPrimCustom)) {
    return nullptr;
  }

  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  bool is_ascend = (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice);
  auto is_ge = (AnfAlgo::GetBackend(func_graph) == kBackendGE);
  auto func_type = common::AnfAlgo::GetNodeAttr<std::string>(cnode, kAttrFuncType);
  // For AOT custom nodes on Ascend (non-GE) all kernel arguments are passed as inputs;
  // No attributes remain to be converted. Skip the Input-to-Attr pass.
  if (func_type == kAOTFuncType && is_ascend && !is_ge) {
    return nullptr;
  }

  mindspore::HashSet<size_t> attr_indices;
  GetCustomOpAttrIndex(common::AnfAlgo::GetCNodePrimitive(cnode), &attr_indices);
  if (attr_indices.empty()) {
    return nullptr;
  }

  return ConstInputToAttr(cnode, attr_indices);
}
}  // namespace opt
}  // namespace mindspore
