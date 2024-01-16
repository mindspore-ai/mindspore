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

#include <limits>
#include "backend/common/pass/add_attr_to_node/add_attr_to_node_register.h"

namespace mindspore {
namespace opt {
const AnfNodePtr NanToNumFusionProcess(const FuncGraphPtr &graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto x_node = cnode->input(kIndex1);
  MS_EXCEPTION_IF_NULL(x_node);
  auto x_abs = x_node->abstract();
  MS_EXCEPTION_IF_NULL(x_abs);
  auto x_dtype = x_abs->BuildType();
  MS_EXCEPTION_IF_NULL(x_dtype);
  auto dtype = x_dtype->cast<TensorTypePtr>();
  TypeId dtype_id = dtype->element()->type_id();
  const auto &prim = common::AnfAlgo::GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(prim);

  size_t idx0 = ops::GetInputIndexByName(common::AnfAlgo::GetCNodeName(cnode), "nan");
  if (idx0 != SIZE_MAX) {
    auto nan_node = common::AnfAlgo::GetInputNode(cnode, idx0);
    if (!utils::isa<ValueNode>(nan_node)) {
      MS_LOG(INTERNAL_EXCEPTION) << "The expected type is ValueNode, but got " << nan_node->type_name() << ".";
    }
    auto nan_value_node_ptr = nan_node->cast<ValueNodePtr>();
    auto nan_value_ptr = utils::cast<ValuePtr>(nan_value_node_ptr->value());
    if (nan_value_ptr->isa<None>()) {
      nan_node->cast<ValueNodePtr>()->set_value(MakeValue(static_cast<float>(0.0)));
    }
  } else {
    MS_LOG(ERROR) << "For '" << cnode->fullname_with_scope() << "', can't not find input of nan.";
    return cnode;
  }

  size_t idx1 = ops::GetInputIndexByName(common::AnfAlgo::GetCNodeName(cnode), "posinf");
  ValuePtr posinf = MakeValue(std::numeric_limits<float>::max());
  if (dtype_id == kNumberTypeFloat16) {
    posinf = MakeValue(static_cast<float>(std::numeric_limits<float16>::max()));
  }
  if (idx1 != SIZE_MAX) {
    auto posinf_node = common::AnfAlgo::GetInputNode(cnode, idx1);
    if (!utils::isa<ValueNode>(posinf_node)) {
      MS_LOG(INTERNAL_EXCEPTION) << "The expected type is ValueNode, but got " << posinf_node->type_name() << ".";
    }
    auto posinf_value_node_ptr = posinf_node->cast<ValueNodePtr>();
    auto posinf_value_ptr = utils::cast<ValuePtr>(posinf_value_node_ptr->value());
    if (posinf_value_ptr->isa<None>()) {
      posinf_node->cast<ValueNodePtr>()->set_value(posinf);
    }
  } else {
    MS_LOG(ERROR) << "For '" << cnode->fullname_with_scope() << "', can't not find input of posinf.";
    return cnode;
  }

  size_t idx2 = ops::GetInputIndexByName(common::AnfAlgo::GetCNodeName(cnode), "neginf");
  ValuePtr neginf = MakeValue(std::numeric_limits<float>::lowest());
  if (dtype_id == kNumberTypeFloat16) {
    neginf = MakeValue(static_cast<float>(std::numeric_limits<float16>::lowest()));
  }
  if (idx2 != SIZE_MAX) {
    auto neginf_node = common::AnfAlgo::GetInputNode(cnode, idx2);
    if (!utils::isa<ValueNode>(neginf_node)) {
      MS_LOG(INTERNAL_EXCEPTION) << "The expected type is ValueNode, but got " << neginf_node->type_name() << ".";
    }
    auto neginf_value_node_ptr = neginf_node->cast<ValueNodePtr>();
    auto neginf_value_ptr = utils::cast<ValuePtr>(neginf_value_node_ptr->value());
    if (neginf_value_ptr->isa<None>()) {
      neginf_node->cast<ValueNodePtr>()->set_value(neginf);
    }
  } else {
    MS_LOG(ERROR) << "For '" << cnode->fullname_with_scope() << "', can't not find input of neginf.";
    return cnode;
  }

  return cnode;
}
}  // namespace opt
}  // namespace mindspore
