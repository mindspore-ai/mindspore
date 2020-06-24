/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_OPTIMIZER_IRPASS_MARK_INTERFACE_FUSION_H
#define MINDSPORE_CCSRC_OPTIMIZER_IRPASS_MARK_INTERFACE_FUSION_H

#include <string>
#include <sstream>
#include <unordered_map>

#include "session/anf_runtime_algorithm.h"
#include "optimizer/optimizer.h"
#include "optimizer/irpass.h"
#include "ir/visitor.h"
#include "operator/ops.h"
#include "utils/graph_utils.h"
#include "operator/composite/composite.h"

namespace mindspore {
namespace opt {
namespace irpass {

static int count = 0;

std::string GetFusionNumber() {
  std::stringstream ss;
  ss << std::setw(4) << std::setfill('0') << count;
  std::string num = ss.str();
  ++count;

  return "_" + num;
}

// Mark CNodes which can be merged in kernel build
class MarkInterfaceFusion : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    if (node->func_graph()->has_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL) && IsPrimitiveCNode(node, prim::kPrimSelect)) {
      auto cnode = node->cast<CNodePtr>();
      auto condition = cnode->input(1);
      std::string cmp;
      std::unordered_map<std::string, std::string> cmp_list = {{"GreaterEqual", "GE"}, {"Greater", "GT"},
                                                               {"LessEqual", "LE"},    {"Less", "LT"},
                                                               {"Equal", "EQ"},        {"NotEqual", "NE"}};
      if (IsPrimitiveCNode(condition)) {
        auto prim_name = GetCNodeFuncName(condition->cast<CNodePtr>());
        if (cmp_list.count(prim_name) != 0) {
          // Mark Select and compare node
          cmp = cmp_list[prim_name];
          auto cnt = GetFusionNumber();
          AnfAlgo::SetNodeAttr("fusion", MakeValue("Select" + cmp + cnt), condition);
          AnfAlgo::SetNodeAttr("fusion", MakeValue("Select" + cmp + cnt + "_end"), node);
          for (size_t i = 1; i < cnode->inputs().size(); ++i) {
            if (IsPrimitiveCNode(cnode->input(i), prim::kPrimZerosLike)) {
              AnfAlgo::SetNodeAttr("fusion", MakeValue("Select" + cmp + cnt), cnode->input(i));
            }
          }
        }
      }
    }
    return nullptr;
  }

  void Visit(const AnfNodePtr &) override {}

 private:
  AnfNodePtr y_{nullptr};
};

}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_OPTIMIZER_IRPASS_MARK_INTERFACE_FUSION_H
