/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_BPROP_GET_MULTITYPE_OPS_H
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_BPROP_GET_MULTITYPE_OPS_H

#include <utility>
#include <string>
#include <vector>
#include <algorithm>
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/anf_visitor.h"

namespace mindspore {
namespace opt {
namespace irpass {
class BpropGetMultitypeOps : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    static mindspore::HashMap<std::string, std::pair<std::string, std::string>> multitype_ops{
      {"S-Prim-zeros_like_leaf", {"zeros_like", ""}},
      {"S-Prim-getitem", {"getitem", "mindspore.ops.composite.multitype_ops.getitem_impl"}},
      {"S-Prim-negative", {"negative", "mindspore.ops.composite.multitype_ops.negative_impl"}},
      {"S-Prim-mul", {"mul", "mindspore.ops.composite.multitype_ops.mul_impl"}},
      {"S-Prim-logical_not", {"logical_not", "mindspore.ops.composite.multitype_ops.logic_not_impl"}},
      {"S-Prim-in", {"in_", "mindspore.ops.composite.multitype_ops.in_impl"}},
      {"S-Prim-less", {"less", "mindspore.ops.composite.multitype_ops.less_impl"}},
      {"S-Prim-add", {"add", "mindspore.ops.composite.multitype_ops.add_impl"}},
    };
    auto fg = node->func_graph();
    if (fg == nullptr) {
      return nullptr;
    }
    auto cnode = node->cast_ptr<CNode>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto &inputs = cnode->inputs();
    if (inputs.empty()) {
      return nullptr;
    }
    auto prim = GetValueNode<PrimitivePtr>(inputs[0]);
    MS_EXCEPTION_IF_NULL(prim);
    auto iter = multitype_ops.find(prim->name());
    if (iter == multitype_ops.end()) {
      return nullptr;
    }
    ValuePtr python_ops;
    if (!iter->second.second.empty()) {
      python_ops = prim::GetPythonOps(iter->second.first, iter->second.second);
    } else {
      python_ops = prim::GetPythonOps(iter->second.first);
    }
    std::vector<AnfNodePtr> new_inputs{NewValueNode(python_ops)};
    (void)std::copy(inputs.begin() + 1, inputs.end(), std::back_inserter(new_inputs));
    return fg->NewCNode(new_inputs);
  }
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_BPROP_GET_MULTITYPE_OPS_H
