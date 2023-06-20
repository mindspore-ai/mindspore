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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_PYNATIVE_NO_GRAD_ELIMINATE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_PYNATIVE_NO_GRAD_ELIMINATE_H_

#include <vector>
#include <algorithm>
#include <string>
#include "frontend/optimizer/irpass.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"

namespace mindspore {
namespace opt {
namespace irpass {
class PynativeNoGradEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) override {
    if (!IsNeedOptimize(optimizer, node) || !node->isa<CNode>()) {
      return nullptr;
    }

    const auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    const auto &node_inputs = cnode->inputs();
    if (need_grad_flag_of_inputs_.size() != node_inputs.size() - 1) {
      return nullptr;
    }

    MS_EXCEPTION_IF_NULL(func_graph_);
    const auto &graph_inputs = func_graph_->get_inputs();
    if (graph_inputs.size() < node_inputs.size() - 1) {
      return nullptr;
    }

    auto manager = optimizer->manager();
    MS_EXCEPTION_IF_NULL(manager);
    for (size_t i = 1; i < node_inputs.size(); ++i) {
      if (!need_grad_flag_of_inputs_[i - 1] && node_inputs[i]->isa<CNode>() &&
          !IsPrimitiveCNode(node_inputs[i], prim::kPrimZerosLike)) {
        const auto &graph_input_type = graph_inputs[i - 1]->Type();
        if (graph_input_type == nullptr || !graph_input_type->isa<TensorType>()) {
          // If input is not tensor, it can not be input for kPrimZerosLike.
          continue;
        }
        AnfNodePtrList new_inputs = {NewValueNode(prim::kPrimZerosLike), graph_inputs[i - 1]};
        auto zeros_like_node = node->func_graph()->NewCNode(new_inputs);
        MS_EXCEPTION_IF_NULL(zeros_like_node);

        zeros_like_node->set_abstract(graph_inputs[i - 1]->abstract());
        if (!manager->Replace(node_inputs[i], zeros_like_node)) {
          MS_LOG(INTERNAL_EXCEPTION) << node_inputs[i]->DebugString() << ", replace node failed.";
        }
      }
    }

    return node;
  }

 private:
  bool IsNeedOptimize(const OptimizerPtr &optimizer, const AnfNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    if (!IsPrimitiveCNode(node, prim::kPrimMakeTuple)) {
      return false;
    }

    MS_EXCEPTION_IF_NULL(optimizer);
    const auto &resource = std::dynamic_pointer_cast<pipeline::Resource>(optimizer->resource());
    MS_EXCEPTION_IF_NULL(resource);

    func_graph_ = resource->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph_);
    if (!func_graph_->has_attr(kAttrNeedGradFlagOfInputs)) {
      return false;
    }

    const size_t ret_input_size = 2;
    const auto &return_node = func_graph_->get_return();
    MS_EXCEPTION_IF_NULL(return_node);
    if (return_node->size() != ret_input_size) {
      // ret node has two input 1 ret op + 1 value
      return false;
    }

    if (return_node->input(1) != node) {
      // Only optimiz return maketuple node.
      return false;
    }

    need_grad_flag_of_inputs_ = GetValue<std::vector<bool>>(func_graph_->get_attr(kAttrNeedGradFlagOfInputs));
    return true;
  }

  std::vector<bool> need_grad_flag_of_inputs_;
  FuncGraphPtr func_graph_;
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_TILE_ELIMINATE_H_
