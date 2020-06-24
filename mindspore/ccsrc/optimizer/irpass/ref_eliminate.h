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
#include "ir/pattern_matcher.h"

namespace mindspore {
namespace opt {
namespace irpass {
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
// {prim::kPrimGetRefOrigin, Parameter} -> Parameter
class GetRefParamEliminater : public OptimizerCaller {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    PatternNode<AnfNodePtr> x;
    MATCH_REPLACE_IF(node, PPrimitive(prim::kPrimGetRefValue, x), x, x.CheckFunc(IsParam, node));
    MATCH_REPLACE_IF(node, PPrimitive(prim::kPrimGetRefOrigin, x), x, x.CheckFunc(IsParam, node));
    return nullptr;
  }
};

// {prim::kPrimGetRefKey, {prim::kPrimMakeRef, X, Y, Z}} -> X
// {prim::kPrimGetRefValue, {prim::kPrimMakeRef, X, Y, Z}} -> Y
// {prim::kPrimGetRefOrigin, {prim::kPrimMakeRef, X, Y, Z}} -> Z
class GetMakeRefEliminater : public OptimizerCaller {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    PatternNode<AnfNodePtr> x, y, z;
    MATCH_REPLACE(node, PPrimitive(prim::kPrimGetRefKey, PPrimitive(prim::kPrimMakeRef, x, y, z)), x);
    MATCH_REPLACE(node, PPrimitive(prim::kPrimGetRefValue, PPrimitive(prim::kPrimMakeRef, x, y, z)), y);
    MATCH_REPLACE(node, PPrimitive(prim::kPrimGetRefOrigin, PPrimitive(prim::kPrimMakeRef, x, y, z)), z);
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
#endif  // MINDSPORE_CCSRC_OPTIMIZER_IRPASS_REF_ELIMINATE_H_
