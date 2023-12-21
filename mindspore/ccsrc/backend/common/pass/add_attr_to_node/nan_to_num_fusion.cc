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

  auto nan_none = prim->GetAttr("nan_none");
  if (nan_none != nullptr && GetValue<bool>(nan_none)) {
    common::AnfAlgo::SetNodeAttr("nan", MakeValue(static_cast<float>(0.0)), cnode);
    size_t idx = ops::GetInputIndexByName(common::AnfAlgo::GetCNodeName(cnode), "nan");
    if (idx != SIZE_MAX) {
      auto nan_node = common::AnfAlgo::GetInputNode(cnode, idx);
      if (utils::isa<ValueNodePtr>(nan_node)) {
        nan_node->cast<ValueNodePtr>()->set_value(MakeValue(static_cast<float>(0.0)));
      }
    }
  }
  auto posinf_none = prim->GetAttr("posinf_none");
  if (posinf_none != nullptr && GetValue<bool>(posinf_none)) {
    size_t idx = ops::GetInputIndexByName(common::AnfAlgo::GetCNodeName(cnode), "posinf");
    ValuePtr posinf = MakeValue(std::numeric_limits<float>::max());
    if (dtype_id == kNumberTypeFloat32) {
      common::AnfAlgo::SetNodeAttr("posinf", MakeValue(std::numeric_limits<float>::max()), cnode);
    } else if (dtype_id == kNumberTypeFloat16) {
      common::AnfAlgo::SetNodeAttr("posinf", MakeValue(static_cast<float>(std::numeric_limits<float16>::max())), cnode);
      posinf = MakeValue(static_cast<float>(std::numeric_limits<float16>::max()));
    }
    if (idx != SIZE_MAX) {
      auto posinf_node = common::AnfAlgo::GetInputNode(cnode, idx);
      if (utils::isa<ValueNodePtr>(posinf_node)) {
        posinf_node->cast<ValueNodePtr>()->set_value(posinf);
      }
    }
  }
  auto neginf_none = prim->GetAttr("neginf_none");
  if (neginf_none != nullptr && GetValue<bool>(neginf_none)) {
    size_t idx = ops::GetInputIndexByName(common::AnfAlgo::GetCNodeName(cnode), "neginf");
    ValuePtr neginf = MakeValue(std::numeric_limits<float>::lowest());
    if (dtype_id == kNumberTypeFloat32) {
      common::AnfAlgo::SetNodeAttr("neginf", MakeValue(std::numeric_limits<float>::lowest()), cnode);
    } else if (dtype_id == kNumberTypeFloat16) {
      common::AnfAlgo::SetNodeAttr("neginf", MakeValue(static_cast<float>(std::numeric_limits<float16>::lowest())),
                                   cnode);
      neginf = MakeValue(static_cast<float>(std::numeric_limits<float16>::lowest()));
    }
    if (idx != SIZE_MAX) {
      auto neginf_node = common::AnfAlgo::GetInputNode(cnode, idx);
      if (utils::isa<ValueNodePtr>(neginf_node)) {
        neginf_node->cast<ValueNodePtr>()->set_value(neginf);
      }
    }
  }

  return cnode;
}
}  // namespace opt
}  // namespace mindspore
