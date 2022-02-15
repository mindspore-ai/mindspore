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

#include "frontend/optimizer/irpass/ge_specialized_prepare.h"

#include <memory>
#include <utility>
#include <unordered_map>

#include "ir/func_graph.h"
#include "frontend/operator/ops.h"

namespace mindspore {
namespace opt {
namespace irpass {
void GeTensorArrayPrepare::InsertFlowOutputToTA(const std::vector<AnfNodePtr> &all_nodes) {
  FuncGraphPtr root = nullptr;
  if (all_nodes.size() == 0) {
    return;
  } else {
    root = all_nodes[0]->func_graph();
  }

  for (auto &ta_input_node : all_nodes) {
    if (!ta_input_node->isa<CNode>()) {
      continue;
    }
    auto ta_input_cnode = ta_input_node->cast<CNodePtr>();
    for (size_t input_index = 0; input_index < ta_input_cnode->inputs().size(); input_index++) {
      auto ta_node = ta_input_cnode->input(input_index);
      if (IsPrimitiveCNode(ta_node, prim::kPrimTensorArray)) {
        auto ta_find = converted_ta_node_.find(ta_node);
        // cached TensorArray node
        if (ta_find != converted_ta_node_.end()) {
          auto new_ta_input_node_input = ta_find->second;
          ta_input_cnode->set_input(input_index, new_ta_input_node_input);
        } else {
          // new a TupleGetItem node and set it's input with TensorArray node and ValueNode(0)
          // set TAInput node input with TupleGetItem node
          int64_t index = 0;

          auto index_value_node = NewValueNode(index);
          auto index_node_abstract = std::make_shared<abstract::AbstractScalar>(index);
          index_value_node->set_abstract(index_node_abstract);

          auto new_tuple_get_cnode = root->NewCNode({NewValueNode(prim::kPrimTupleGetItem), ta_node, index_value_node});
          auto new_tuple_get_node = new_tuple_get_cnode->cast<AnfNodePtr>();

          auto tuple_get_node_abstract = ta_node_abstract_cache_[ta_node];
          new_tuple_get_node->set_abstract(tuple_get_node_abstract);

          converted_ta_node_[ta_node] = new_tuple_get_node;
          ta_input_cnode->set_input(input_index, new_tuple_get_node);
        }
      }
    }
  }
}

void GeTensorArrayPrepare::TransformTASizeFromAttrToInput(const AnfNodePtr &node) {
  auto ta_node = node->cast<CNodePtr>();
  int32_t res_size = 0;
  PrimitivePtr prim = GetValueNode<PrimitivePtr>(ta_node->input(0));
  // get size attr
  if (prim->HasAttr("size")) {
    auto size_value_ptr = prim->GetAttr("size");
    auto size = GetValue<int64_t>(size_value_ptr);
    res_size = static_cast<int32_t>(size);
  }
  // generate size input
  auto size_node = NewValueNode(MakeValue(res_size));
  auto node_abstract = std::make_shared<abstract::AbstractScalar>(res_size);
  size_node->set_abstract(node_abstract);
  auto origin_inputs = ta_node->inputs();
  // set cnode input
  ta_node->add_input(size_node);
  // has monad input
  if (origin_inputs.size() > 1) {
    std::vector<AnfNodePtr> sorted_inputs(origin_inputs);
    sorted_inputs.insert(sorted_inputs.begin() + 1, size_node);
    ta_node->set_inputs(sorted_inputs);
  }

  // get origin abstract
  auto origin_ta_abstract = ta_node->abstract();
  // new tuple abstract
  std::vector<AbstractBasePtr> abstract_list;
  // push origin abstract
  abstract_list.push_back(origin_ta_abstract);
  // new flow abstract
  float flow_value = 0.0;
  auto flow_abstract = std::make_shared<abstract::AbstractScalar>(flow_value);
  // push flow abstract
  abstract_list.push_back(flow_abstract);
  // cache TensorArray node's abstract
  auto abstract_find = ta_node_abstract_cache_.find(ta_node);
  if (abstract_find == ta_node_abstract_cache_.end()) {
    ta_node_abstract_cache_[ta_node] = ta_node->abstract();
  }
  // modify TensorArray node output's abstract from Tensor to Tuple
  auto new_ta_abstract = std::make_shared<abstract::AbstractTuple>(abstract_list);
  ta_node->set_abstract(new_ta_abstract);
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
