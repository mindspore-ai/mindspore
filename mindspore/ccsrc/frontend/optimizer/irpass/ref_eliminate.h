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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_REF_ELIMINATE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_REF_ELIMINATE_H_

#include <memory>

#include "ir/pattern_matcher.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
namespace irpass {
namespace internal {
class GetRefValueTransform {
 public:
  GetRefValueTransform() {}
  ~GetRefValueTransform() = default;

  AnfNodePtr operator()(const AnfNodePtr &node) {
    CNodePtr cnode = node->cast<CNodePtr>();
    auto inputs = cnode->inputs();
    auto fg = GetValueNode(inputs[0])->cast<FuncGraphPtr>();
    if (fg != nullptr && fg->recursive()) {
      MS_LOG(DEBUG) << "Get refvalue by pass recursive:" << fg->ToString();
      return node;
    }
    auto new_fg = TransformableClone(fg, std::make_shared<TraceTransform>("GetRefValue"));
    auto output = new_fg->output();
    new_fg->set_output(new_fg->NewCNode({NewValueNode(prim::kPrimGetRefValue), output}));
    inputs[0] = NewValueNode(new_fg);
    auto ret_node = cnode->func_graph()->NewCNode(inputs);
    return ret_node;
  }
};
}  // namespace internal

// {prim::kPrimMakeRef, X, Y, Z} -> Y
class MakeRefEliminater : public OptimizerCaller {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    PatternNode<AnfNodePtr> x, y, z;
    MATCH_REPLACE(node, PPrimitive(prim::kPrimMakeRef, x, y, z), y);
    return nullptr;
  }
};

// {prim::kPrimGetRefValue, Parameter} -> Parameter
class GetRefParamEliminater : public OptimizerCaller {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    PatternNode<AnfNodePtr> x;
    MATCH_REPLACE(node, PPrimitive(prim::kPrimGetRefValue, x), x);
    return nullptr;
  }
};

// {prim::kPrimGetRefKey, {prim::kPrimMakeRef, X, Y, Z}} -> X
// {prim::kPrimGetRefValue, {prim::kPrimMakeRef, X, Y, Z}} -> Y
// {prim::kPrimGetRefValue, {prim::switch, cond, t, f}} -> {prim::switch, cond, t, f}
class GetMakeRefEliminater : public OptimizerCaller {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    PatternNode<AnfNodePtr> x, y, z;
    MATCH_REPLACE(node, PPrimitive(prim::kPrimGetRefKey, PPrimitive(prim::kPrimMakeRef, x, y, z)), x);
    MATCH_REPLACE(node, PPrimitive(prim::kPrimGetRefValue, PPrimitive(prim::kPrimMakeRef, x, y, z)), y);
    MATCH_REPLACE_IF(node, PPrimitive(prim::kPrimGetRefValue, x), x, x.CheckFunc(IsCNodeSwitch, node));
    internal::GetRefValueTransform trans;
    auto GetRefLambda = [&trans, &x, &node]() -> AnfNodePtr {
      auto rep = trans(x.GetNode(node));
      if (rep != nullptr) {
        return rep;
      }
      return nullptr;
    };
    MATCH_REPLACE_LAMBDA_IF(node, PPrimitive(prim::kPrimGetRefValue, x), GetRefLambda, x.CheckFunc(IsCNodeGraph, node));
    return nullptr;
  }
};

// IsValueNode<RefKey>
class ReplaceRefkeyByParam : public OptimizerCaller {
 public:
  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) override {
    auto RefKeyLambda = [&node, &optimizer]() -> AnfNodePtr {
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
    };
    PatternNode<AnfNodePtr> x;
    MATCH_REPLACE_LAMBDA_IF(node, x, RefKeyLambda, x.CheckFunc(IsValueNode<RefKey>, node));
    return nullptr;
  }
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_REF_ELIMINATE_H_
