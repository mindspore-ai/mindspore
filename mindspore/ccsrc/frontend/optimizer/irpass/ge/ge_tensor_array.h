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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_GE_TENSOR_ARRAY_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_GE_TENSOR_ARRAY_H_

#include <vector>
#include <memory>
#include <algorithm>

#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"

namespace mindspore {
namespace opt {
namespace irpass {
class GeTensorArrayAddFlow : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimTensorArrayWrite, {IsNode, IsNode, IsNode, IsNode})(node);
    AnfVisitor::Match(prim::kPrimTensorArrayGather, {IsNode, IsNode, IsNode})(node);

    // Check if the pattern matches.
    if (!is_match_ || node->func_graph() == nullptr) {
      return nullptr;
    }

    auto ta_node = node->cast<CNodePtr>();
    float flow_value = 0.0;
    // generate flow input
    auto flow_node = NewValueNode(MakeValue(flow_value));
    // set abstract
    auto node_abstract = std::make_shared<abstract::AbstractScalar>(flow_value);
    flow_node->set_abstract(node_abstract);
    // add cnode input
    auto ta_node_inputs = ta_node->inputs();
    if (HasAbstractMonad(ta_node_inputs.back())) {
      auto input_size = ta_node_inputs.size();
      std::vector<AnfNodePtr> new_inputs;
      new_inputs.assign(ta_node_inputs.begin(), ta_node_inputs.end());
      new_inputs.insert(new_inputs.begin() + input_size - 1, flow_node);
      ta_node->set_inputs(new_inputs);
    } else {
      ta_node->add_input(flow_node);
    }
    return ta_node;
  }

  void Visit(const AnfNodePtr &node) override { is_match_ = true; }

  void Reset() { is_match_ = false; }

 private:
  bool is_match_{false};
};

class GeTensorArrayCastIndex : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimTensorArrayWrite, {IsNode, IsNode, IsNode, IsNode, IsNode})(node);

    // Check if the pattern matches.
    if (!is_match_ || node->func_graph() == nullptr) {
      return nullptr;
    }

    const size_t index_input_index = 2;
    auto index_input_node = node->cast<CNodePtr>()->input(index_input_index);
    // Get cast prim
    auto cast_primitive = std::make_shared<Primitive>(prim::kPrimCast->name());

    TypePtr src_type = TypeIdToType(TypeId::kNumberTypeInt64);
    TypePtr dst_type = TypeIdToType(TypeId::kNumberTypeInt32);
    auto src_attr_value = MakeValue(src_type);
    auto dst_attr_value = MakeValue(dst_type);
    auto prim = std::make_shared<Primitive>(cast_primitive->AddAttr("dst_type", dst_attr_value));
    prim = std::make_shared<Primitive>(prim->AddAttr("DstT", dst_attr_value));
    prim = std::make_shared<Primitive>(prim->AddAttr("SrcT", src_attr_value));

    // Insert cast
    auto type_node = NewValueNode(dst_type);
    type_node->set_abstract(dst_type->ToAbstract());

    auto new_node = node->func_graph()->NewCNode({NewValueNode(prim), index_input_node, type_node});
    auto cast_abstract = index_input_node->abstract();
    cast_abstract->set_type(dst_type);
    new_node->set_abstract(cast_abstract);

    auto cnode = node->cast<CNodePtr>();
    cnode->set_input(index_input_index, new_node);
    return node;
  }

  void Visit(const AnfNodePtr &node) override { is_match_ = true; }

  void Reset() { is_match_ = false; }

 private:
  bool is_match_{false};
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_GE_TENSOR_ARRAY_H_
