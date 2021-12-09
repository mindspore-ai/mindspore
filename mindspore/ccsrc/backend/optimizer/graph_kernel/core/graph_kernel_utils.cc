/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "backend/optimizer/graph_kernel/core/graph_kernel_utils.h"
#include <algorithm>
#include <memory>
#include <sstream>
#include "base/core_ops.h"
#include "utils/anf_utils.h"
#include "utils/utils.h"
#include "backend/optimizer/graph_kernel/core/graph_kernel_callback.h"

namespace mindspore::graphkernel {
std::string GkUtils::ExtractGraphKernelName(const AnfNodePtrList &nodes, const std::string &prefix,
                                            const std::string &postfix) {
  std::stringstream name;
  if (!prefix.empty()) {
    name << prefix << "_";
  }
  for (const auto &node : nodes) {
    if (AnfUtils::IsGraphKernel(node)) {
      auto fg_flag_val = GetCNodeFuncGraph(node)->get_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL);
      name << GetValue<std::string>(fg_flag_val) << "_";
    } else if (node->isa<CNode>() && AnfUtils::IsRealKernel(node)) {
      name << GetCNodePrimitive(node)->name() << "_";
    }
  }
  if (!postfix.empty()) {
    name << postfix;
  }
  return name.str();
}

AnfNodePtrList GkUtils::SpreadTuples(const AnfNodePtrList &nodes, size_t begin_index) {
  AnfNodePtrList result;
  for (size_t i = begin_index; i < nodes.size(); i++) {
    if (IsPrimitiveCNode(nodes[i], prim::kPrimMakeTuple)) {
      auto mt = nodes[i]->cast<CNodePtr>();
      // recursively spread all inner tuples.
      auto mt_inputs = SpreadTuples(mt->inputs(), 1);
      result.insert(result.end(), mt_inputs.begin(), mt_inputs.end());
    } else {
      result.push_back(nodes[i]);
    }
  }
  return result;
}

std::vector<PrimitivePtr> GkUtils::GetValidOps(const std::vector<OpWithLevel> &ops_with_level, unsigned int level,
                                               const std::vector<std::string> &enable_ops_only,
                                               const std::vector<std::string> &enable_ops,
                                               const std::vector<std::string> &disable_ops) {
  std::vector<PrimitivePtr> ops;
  auto new_prim = [](const std::string &name) { return std::make_shared<Primitive>(name); };
  if (!enable_ops_only.empty()) {
    (void)std::transform(enable_ops_only.begin(), enable_ops_only.end(), std::back_inserter(ops), new_prim);
    return ops;
  }
  auto target = Callback::Instance()->GetTargetFromContext();
  for (const auto &[op_target, op_level, op] : ops_with_level) {
    if (op_target == kAllTarget || op_target == target) {
      if (level >= op_level) {
        (void)ops.emplace_back(op);
      }
    }
  }
  if (!enable_ops.empty()) {
    (void)std::transform(enable_ops.begin(), enable_ops.end(), std::back_inserter(ops), new_prim);
  }
  if (!disable_ops.empty()) {
    auto iter = std::remove_if(ops.begin(), ops.end(), [&disable_ops](const PrimitivePtr &p) {
      return std::find(disable_ops.begin(), disable_ops.end(), p->name()) != disable_ops.end();
    });
    (void)ops.erase(iter, ops.end());
  }
  return ops;
}

bool GkUtils::IsKeepBasicNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto prim = GetCNodePrimitive(node);
  if (prim == nullptr) return false;

  // dynamic shape nodes is not supported yet.
  // the "skip" is used by inplace node.
  // the kAttrIsInternalOutput is used by internal output of KernelGraph.
  const std::vector<std::string> exclude_bool_attrs = {kAttrInputIsDynamicShape, kAttrOutputIsDynamicShape,
                                                       kAttrIsDynamicShape, "skip", kAttrIsInternalOutput};
  if (std::any_of(exclude_bool_attrs.cbegin(), exclude_bool_attrs.cend(), [&prim](const std::string &attr_name) {
        return prim->HasAttr(attr_name) && GetValue<bool>(prim->GetAttr(attr_name));
      })) {
    return true;
  }

  // If node contain attribute in contagious_attrs, it have to keep basic no matter what the value is.
  const std::vector<std::string> contagious_attrs = {"inplace_group", "inplace_algo", "inplace_output_index",
                                                     "aggregate", "aggregate_input_index"};
  if (std::any_of(contagious_attrs.cbegin(), contagious_attrs.cend(),
                  [&prim](const std::string &attr_name) -> bool { return prim->HasAttr(attr_name); })) {
    return true;
  }
  return false;
}
}  // namespace mindspore::graphkernel
