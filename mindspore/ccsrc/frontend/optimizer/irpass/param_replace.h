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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_PARAM_REPLACE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_PARAM_REPLACE_H_

#include <memory>
#include <string>

#include "include/frontend/optimizer/optimizer.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"
#include "frontend/jit/ps/parse/parse.h"

namespace mindspore {
namespace opt {
namespace irpass {
class ReplaceOldParam : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) override {
    if (!IsParam(node)) {
      return nullptr;
    }
    auto resource = std::dynamic_pointer_cast<pipeline::Resource>(optimizer->resource());
    MS_EXCEPTION_IF_NULL(resource);

    auto top_graph = resource->func_graph();  // parse::Parser::GetTopFuncGraph();
    MS_EXCEPTION_IF_NULL(top_graph);

    auto param_node = node->cast<ParameterPtr>();
    if (param_node == nullptr || !param_node->has_default() || node->func_graph() == top_graph) {
      return nullptr;
    }
    auto para_name = param_node->name();
    for (const auto &tnode : top_graph->parameters()) {
      auto para = tnode->cast<ParameterPtr>();
      if (para != nullptr && para->name() == para_name) {
        const auto &param_node_fg = param_node->func_graph();
        const auto &param_node_fg_name = param_node_fg->ToString();
        if (param_node_fg_name.find("_CR_") != string::npos) {
          std::string new_param_name = para_name + "_CR";
          MS_LOG(INFO) << "Fix param_node name from " << para_name << " to " << new_param_name;
          param_node->set_name(new_param_name);
          return nullptr;
        }
        return para;
      }
    }
    return nullptr;
  }
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_PARAM_REPLACE_H_
