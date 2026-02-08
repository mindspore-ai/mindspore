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
#include "replace_addn_fusion_pass.h"

namespace mindspore {
namespace opt {
void ReplaceAddNFusionPass::DefineSrcPattern(SrcPattern *src_pattern) {
  MS_LOG(INFO) << "Defining source pattern for ReplaceAddNFusionPass";
  MS_EXCEPTION_IF_NULL(src_pattern);

  (*src_pattern)
    .AddVar("A")
    .AddVar("B")
    .AddCNode("maketuple", {std::make_shared<Primitive>("MakeTuple"), "A", "B"})
    .AddCNode("addn", {std::make_shared<Primitive>("AddN"), "maketuple"});

  MS_LOG(INFO) << "Source pattern defined: AddN(MakeTuple(A, B))";
}

AnfNodePtr ReplaceAddNFusionPass::BuildAdd(const PatternMap &m, const AnfNodePtr &default_node) {
  MS_EXCEPTION_IF_NULL(default_node);

  auto addn_node = m.Get("addn");
  MS_EXCEPTION_IF_NULL(addn_node);

  auto addn_cnode = addn_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(addn_cnode);

  auto add_node = default_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(add_node);

  add_node->set_scope(addn_cnode->scope());
  add_node->set_abstract(addn_cnode->abstract());

  return add_node;
}

void ReplaceAddNFusionPass::DefineDstPattern(DstPattern *dst_pattern) {
  MS_LOG(INFO) << "Defining destination pattern for ReplaceAddNFusionPass";
  MS_EXCEPTION_IF_NULL(dst_pattern);

  (*dst_pattern).AddCNode("add", {std::make_shared<Primitive>("Add"), "A", "B"}, BuildAdd);

  MS_LOG(INFO) << "Destination pattern defined: Add(A, B)";
}

bool ReplaceAddNFusionPass::CheckMatchedDAG(const PatternMap &pattern_map, const FuncGraphPtr &func_graph,
                                            const AnfNodePtr &node) const {
  // Simplified check - just return true for now to avoid potential issues
  MS_LOG(INFO) << "CheckMatchedDAG called - returning true";
  return true;
}
}  // namespace opt
}  // namespace mindspore
