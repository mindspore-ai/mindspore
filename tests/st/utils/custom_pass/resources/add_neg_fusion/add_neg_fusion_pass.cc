/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "add_neg_fusion_pass.h"

namespace mindspore {
namespace opt {
void AddNegFusionPass::DefineSrcPattern(SrcPattern *src_pattern) {
  MS_LOG(INFO) << "Defining source pattern for AddNegFusionPass";
  MS_EXCEPTION_IF_NULL(src_pattern);

  // Pattern: Add(x, Neg(y))
  (*src_pattern)
    .AddVar("x")
    .AddVar("y")
    .AddCNode("neg", {std::make_shared<Primitive>("Neg"), "y"})
    .AddCNode("add", {std::make_shared<Primitive>("Add"), "x", "neg"});

  MS_LOG(INFO) << "Source pattern defined: Add(x, Neg(y))";
}

AnfNodePtr AddNegFusionPass::BuildSub(const PatternMap &m, const AnfNodePtr &default_node) {
  auto add_node = m.Get("add")->cast<CNodePtr>();
  auto neg_node = m.Get("neg")->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(add_node);
  MS_EXCEPTION_IF_NULL(neg_node);

  auto sub_node = default_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(sub_node);

  // Copy Add node's scope to maintain execution context
  sub_node->set_scope(add_node->scope());

  // Set abstract same as Add output
  auto add_abstract = add_node->abstract();
  if (add_abstract != nullptr) {
    sub_node->set_abstract(add_abstract->Clone());
  } else {
    MS_LOG(EXCEPTION) << "Failed to create Sub abstract from Add node";
  }

  return sub_node;
}

void AddNegFusionPass::DefineDstPattern(DstPattern *dst_pattern) {
  MS_LOG(INFO) << "Defining destination pattern for AddNegFusionPass";
  MS_EXCEPTION_IF_NULL(dst_pattern);

  // Replace with Sub(x, y) - directly subtract y instead of adding its negation
  (*dst_pattern).AddCNode("sub", {std::make_shared<Primitive>("Sub"), "x", "y"}, BuildSub);

  MS_LOG(INFO) << "Destination pattern defined: Sub(x, y)";
}

bool AddNegFusionPass::CheckMatchedDAG(const PatternMap &pattern_map, const FuncGraphPtr &func_graph,
                                       const AnfNodePtr &node) const {
  auto add_node = pattern_map.Get("add");
  if (!add_node) {
    MS_LOG(ERROR) << "Add node not found in pattern match";
    return false;
  }

  auto neg_node = pattern_map.Get("neg");
  if (!neg_node) {
    MS_LOG(ERROR) << "Neg node not found in pattern match";
    return false;
  }

  auto x_node = pattern_map.Get("x");
  if (!x_node) {
    MS_LOG(ERROR) << "x node not found in pattern match";
    return false;
  }

  auto y_node = pattern_map.Get("y");
  if (!y_node) {
    MS_LOG(ERROR) << "y node not found in pattern match";
    return false;
  }

  MS_LOG(INFO) << "AddNeg fusion pattern matched successfully";
  return true;
}
}  // namespace opt
}  // namespace mindspore
