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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_SEQUENCE_TO_SEQUENCE_ELIMINATE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_SEQUENCE_TO_SEQUENCE_ELIMINATE_H_

#include <memory>
#include <vector>

#include "frontend/optimizer/optimizer_caller.h"
#include "mindspore/core/ops/structure_ops.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/optimizer.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace opt {
namespace irpass {
// {prim::kPrimListToTuple, data} => {prim::kPrimMakeTuple, {prim::kPrimListGetItem, data, 0}, ...}
class ListToTupleEliminator : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    AnfVisitor::Match(prim::kPrimListToTuple, {IsCNode})(node);
    auto fg = node->func_graph();
    if (fg != nullptr) {
      std::vector<AnfNodePtr> args_{NewValueNode(prim::kPrimMakeTuple)};
      if (real_node == nullptr) {
        return nullptr;
      }
      auto input_abs = real_node->abstract()->cast<abstract::AbstractListPtr>();
      MS_EXCEPTION_IF_NULL(input_abs);
      if (!input_abs->dynamic_len()) {
        for (size_t i = 0; i < input_abs->size(); ++i) {
          auto item = fg->NewCNode({NewValueNode(prim::kPrimListGetItem), real_node, NewValueNode(SizeToLong(i))});
          item->set_abstract(real_node->abstract());
          args_.emplace_back(item);
        }
        auto new_node = fg->NewCNode(args_);
        new_node->set_abstract(node->abstract());
        return new_node;
      }
    }
    return nullptr;
  }

  void Visit(const CNodePtr &cnode) override {
    real_node = cnode;
    while (IsPrimitiveCNode(real_node, prim::kPrimDepend)) {
      auto depend = real_node->cast<CNodePtr>();
      real_node = depend->input(1)->cast<CNodePtr>();
    }
  }

 private:
  CNodePtr real_node{nullptr};
};

// {prim::kPrimTupleToList, data} => {prim::kPrimMakeList, {prim::kPrimTupleGetItem, data, 0}, ...}
class TupleToListEliminator : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    AnfVisitor::Match(prim::kPrimTupleToList, {IsCNode})(node);
    auto fg = node->func_graph();
    if (fg != nullptr) {
      std::vector<AnfNodePtr> args_{NewValueNode(prim::kPrimMakeList)};
      if (real_node == nullptr) {
        return nullptr;
      }
      auto input_abs = real_node->abstract()->cast<abstract::AbstractTuplePtr>();
      MS_EXCEPTION_IF_NULL(input_abs);
      if (!input_abs->dynamic_len()) {
        for (size_t i = 0; i < input_abs->size(); ++i) {
          auto item = fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), real_node, NewValueNode(SizeToLong(i))});
          item->set_abstract(real_node->abstract());
          args_.emplace_back(item);
        }
        auto new_node = fg->NewCNode(args_);
        new_node->set_abstract(node->abstract());
        return new_node;
      }
    }
    return nullptr;
  }

  void Visit(const CNodePtr &cnode) override {
    real_node = cnode;
    while (IsPrimitiveCNode(real_node, prim::kPrimDepend)) {
      auto depend = real_node->cast<CNodePtr>();
      real_node = depend->input(1)->cast<CNodePtr>();
    }
  }

 private:
  CNodePtr real_node{nullptr};
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_SEQUENCE_TO_SEQUENCE_ELIMINATE_H_
