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

#define USE_DEPRECATED_API
#include "tools/optimizer/fusion/reshape_like_operator_ablation.h"
#include <set>
#include <string>
#include <vector>
#include "nnacl/op_base.h"
#include "include/errorcode.h"

namespace mindspore {
namespace opt {
namespace {
const std::set<std::string> kReshapeLikeOp = {"Reshape", "Squeeze", "Unsqueeze", "ExpandDims"};
bool IsReshapeLikeOp(const AnfNodePtr &anf_node) {
  auto prim = GetCNodePrimitive(anf_node);
  if (prim == nullptr) {
    return false;
  }
  auto op_type = prim->name();
  return kReshapeLikeOp.find(op_type) != kReshapeLikeOp.end();
}
}  // namespace

bool AblateReshapeLikeOp::Run(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto ret = preprocessor_.Run(func_graph);
  if (ret == lite::RET_NOT_SUPPORT) {
    return true;
  }
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Do infershape for dynamic-shape model failed.";
    return ret;
  }
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_MSG(manager != nullptr, lite::RET_NULL_PTR,
                    "FuncGraph's manager is a nullptr, please generate before.");
  auto node_list = TopoSort(func_graph->get_return());
  for (const auto &node : node_list) {
    MS_CHECK_TRUE_MSG(node != nullptr, false, "find an anfNode is a nullptr.");
    if (!utils::isa<CNode>(node)) {
      continue;
    }
    if (!IsReshapeLikeOp(node)) {
      continue;
    }
    if (DoAblation(func_graph, node->cast<CNodePtr>()) != lite::RET_OK) {
      MS_LOG(ERROR) << "Do reshape-ablation failed, node name: " << node->fullname_with_scope();
      return false;
    }
  }
  return true;
}

int AblateReshapeLikeOp::DoAblation(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  const auto &container = preprocessor_.GetShapeContainer();
  auto iter = container.find(cnode);
  if (iter == container.end()) {
    return lite::RET_OK;
  }
  if (iter->second.second.size() != 1) {
    MS_LOG(ERROR) << "Reshape-like op only has one out,but now is " << iter->second.second.size() << ", the node is "
                  << cnode->fullname_with_scope();
    return lite::RET_ERROR;
  }
  const auto &out_shape = iter->second.second.front();
  AnfNodePtr pre_node = cnode->input(1);
  std::vector<CNodePtr> link_ops;
  link_ops.push_back(cnode);
  while (utils::isa<CNode>(pre_node)) {
    auto pre_cnode = pre_node->cast<CNodePtr>();
    if (!IsReshapeLikeOp(pre_cnode)) {
      break;
    }
    link_ops.push_back(pre_cnode);
    pre_node = pre_cnode->input(1);
  }
  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  for (int i = static_cast<int>(link_ops.size()) - 1; i >= 0; --i) {
    iter = container.find(link_ops[i]);
    if (iter == container.end()) {
      continue;
    }
    if (iter->second.first.empty()) {
      MS_LOG(ERROR) << "Reshape-like op has at least one input,but now is " << iter->second.first.size()
                    << ", the node is " << cnode->fullname_with_scope();
      return lite::RET_ERROR;
    }
    const auto &in_shape = iter->second.first.front();
    if (out_shape == in_shape) {
      if (!manager->Replace(cnode, link_ops[i]->input(1))) {
        MS_LOG(ERROR) << "Manager: Replace op failed, old node: " << cnode->fullname_with_scope()
                      << ", new node: " << link_ops[i]->input(1)->fullname_with_scope();
        return lite::RET_ERROR;
      }
      break;
    }
  }
  return lite::RET_OK;
}
}  // namespace opt
}  // namespace mindspore
