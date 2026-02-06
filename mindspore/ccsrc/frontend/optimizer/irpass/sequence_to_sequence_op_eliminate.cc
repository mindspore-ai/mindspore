/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "frontend/optimizer/irpass/sequence_to_sequence_op_eliminate.h"
#include <memory>
#include <vector>

#include "frontend/optimizer/optimizer_caller.h"
#include "primitive/structure_ops.h"
#include "primitive/sequence_ops.h"
#include "primitive/framework_ops.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"
#include "include/frontend/optimizer/optimizer.h"
#include "include/utils/utils.h"

namespace mindspore {
namespace opt {
namespace irpass {
// {prim::kPrimListToTuple, data} => {prim::kPrimMakeTuple, {prim::kPrimListGetItem, data, 0}, ...}

AnfNodePtr ListToTupleEliminator::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimListToTuple)) {
    return nullptr;
  }
  auto fg = node->func_graph();
  if (fg != nullptr) {
    auto real_node = node->cast<CNodePtr>()->input(1);
    MS_EXCEPTION_IF_NULL(real_node);
    std::vector<AnfNodePtr> args_{NewValueNode(prim::kPrimMakeTuple)};
    if (real_node->abstract() == nullptr || !real_node->abstract()->isa<abstract::AbstractList>()) {
      return nullptr;
    }
    auto input_abs = real_node->abstract()->cast<abstract::AbstractListPtr>();
    MS_EXCEPTION_IF_NULL(input_abs);
    if (!input_abs->dynamic_len()) {
      for (size_t i = 0; i < input_abs->size(); ++i) {
        auto item = fg->NewCNode({NewValueNode(prim::kPrimListGetItem), real_node, NewValueNode(SizeToLong(i))});
        item->set_abstract(real_node->abstract());
        (void)args_.emplace_back(item);
      }
      auto new_node = fg->NewCNode(args_);
      new_node->set_abstract(node->abstract());
      return new_node;
    }
  }
  return nullptr;
}

// {prim::kPrimTupleToList, data} => {prim::kPrimMakeList, {prim::kPrimTupleGetItem, data, 0}, ...}
AnfNodePtr TupleToListEliminator::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimTupleToList)) {
    return nullptr;
  }
  auto fg = node->func_graph();
  if (fg != nullptr) {
    auto real_node = node->cast<CNodePtr>()->input(1);
    MS_EXCEPTION_IF_NULL(real_node);
    std::vector<AnfNodePtr> args_{NewValueNode(prim::kPrimMakeList)};
    MS_EXCEPTION_IF_NULL(real_node->abstract());
    auto input_abs = real_node->abstract()->cast<abstract::AbstractTuplePtr>();
    MS_EXCEPTION_IF_NULL(input_abs);
    if (!input_abs->dynamic_len()) {
      for (size_t i = 0; i < input_abs->size(); ++i) {
        auto item = fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), real_node, NewValueNode(SizeToLong(i))});
        item->set_abstract(real_node->abstract());
        (void)args_.emplace_back(item);
      }
      auto new_node = fg->NewCNode(args_);
      new_node->set_abstract(node->abstract());
      return new_node;
    }
  }
  return nullptr;
}

}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
