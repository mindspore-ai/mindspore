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

#ifndef MINDSPORE_CCSRC_OPTIMIZER_IRPASS_REF_ELIMINATE_H_
#define MINDSPORE_CCSRC_OPTIMIZER_IRPASS_REF_ELIMINATE_H_

#include <memory>

#include "optimizer/optimizer.h"
#include "optimizer/irpass.h"
#include "ir/visitor.h"
#include "operator/ops.h"
#include "utils/graph_utils.h"
#include "operator/composite/composite.h"

namespace mindspore {
namespace opt {
namespace irpass {
// {prim::kPrimMakeRef, X, Y, Z} -> Y
class MakeRefEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    y_ = nullptr;
    auto gety = [this](const AnfNodePtr &node) -> bool {
      this->y_ = node;
      return true;
    };

    AnfVisitor::Match(prim::kPrimMakeRef, {IsNode, gety, IsNode})(node);
    return y_;
  }

  void Visit(const AnfNodePtr &) override {}

 private:
  AnfNodePtr y_{nullptr};
};

// {prim::kPrimGetRefValue, Parameter} -> Parameter
// {prim::kPrimGetRefOrigin, Parameter} -> Parameter
class GetRefParamEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    x_ = nullptr;
    AnfVisitor::Match(prim::kPrimGetRefOrigin, {IsParam})(node);
    if (x_ != nullptr) {
      return x_;
    }
    AnfVisitor::Match(prim::kPrimGetRefValue, {IsParam})(node);
    return x_;
  }

  void Visit(const AnfNodePtr &node) override { x_ = node; }

 private:
  AnfNodePtr x_{nullptr};
};

// {prim::kPrimGetRefKey, {prim::kPrimMakeRef, X, Y, Z}} -> X
// {prim::kPrimGetRefValue, {prim::kPrimMakeRef, X, Y, Z}} -> Y
// {prim::kPrimGetRefOrigin, {prim::kPrimMakeRef, X, Y, Z}} -> Z
class GetMakeRefEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr || cnode->size() != 2) {
      return nullptr;
    }

    // {prim::kPrimGetRefKey/Value, {...}}
    auto ref = cnode->input(1)->cast<CNodePtr>();
    if (ref == nullptr || !ref->IsApply(prim::kPrimMakeRef) || ref->size() != 4) {
      return nullptr;
    }

    // {prim::kPrimMakeRef, X, Y, Z}
    if (cnode->IsApply(prim::kPrimGetRefKey)) {
      return ref->input(1);
    }

    if (cnode->IsApply(prim::kPrimGetRefValue)) {
      return ref->input(2);
    }

    if (cnode->IsApply(prim::kPrimGetRefOrigin)) {
      return ref->input(3);
    }

    return nullptr;
  }
};

// IsValueNode<RefKey>
class ReplaceRefkeyByParam : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) override {
    if (!IsValueNode<RefKey>(node)) {
      return nullptr;
    }

    auto refkey = GetValueNode<RefKeyPtr>(node);
    auto resource = std::dynamic_pointer_cast<pipeline::Resource>(optimizer->resource());
    MS_EXCEPTION_IF_NULL(resource);

    auto top_graph = resource->func_graph();
    MS_EXCEPTION_IF_NULL(top_graph);

    for (const auto &tnode : top_graph->parameters()) {
      auto para = tnode->cast<ParameterPtr>();
      if (para != nullptr && para->name() == refkey->tag()) {
        return para;
      }
    }
    return nullptr;
  }
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_OPTIMIZER_IRPASS_REF_ELIMINATE_H_
