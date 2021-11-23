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
#include <sstream>
#include "base/core_ops.h"
#include "utils/anf_utils.h"

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
}  // namespace mindspore::graphkernel
