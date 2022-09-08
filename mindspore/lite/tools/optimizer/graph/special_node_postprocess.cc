/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "tools/optimizer/graph/special_node_postprocess.h"
#include <memory>
#include <utility>
#include <vector>
#include "include/errorcode.h"
#include "tools/optimizer/common/format_utils.h"
#include "nnacl//op_base.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace opt {
namespace {
const PrimitivePtr kPrimInstanceNorm = std::make_shared<Primitive>("InstanceNorm");
ShapeVector GenerateNewShape(const abstract::AbstractBasePtr &abstract) {
  MS_ASSERT(abstract != nullptr);
  ShapeVector shape;
  if (FetchShapeFromAbstract(abstract, &shape) != lite::RET_OK) {
    return shape;
  }
  if (shape.size() == kInputSizeFour) {
    ShapeVector real_shape = {shape[0], shape[kInputIndexThree], shape[1], shape[kInputIndexTwo]};
    shape = real_shape;
  }
  return shape;
}
}  // namespace

bool SpecialNodePostProcess::Run(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto manager = Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr.";
    return false;
  }
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNode>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (CheckPrimitiveType(cnode, prim::kPrimIf) || CheckPrimitiveType(cnode, prim::kPrimWhile)) {
      auto sub_func_graph = GetValueNode<FuncGraphPtr>(cnode->input(1));
      if (sub_func_graph == nullptr) {
        lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
        return false;
      }
      if (!Run(sub_func_graph)) {
        MS_LOG(ERROR) << "postprocess for handling special node failed.";
        return false;
      }
      if (sub_func_graph == nullptr) {
        lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
        return false;
      }
      if (!Run(sub_func_graph)) {
        MS_LOG(ERROR) << "postprocess for handling special node failed.";
        return false;
      }
      continue;
    }
    if (!CheckInstanceNorm(func_graph, cnode)) {
      continue;
    }
    if (HandleInstanceNorm(func_graph, cnode) != lite::RET_OK) {
      MS_LOG(ERROR) << "post-process instance_norm failed.";
      return false;
    }
  }
  return true;
}

bool SpecialNodePostProcess::CheckInstanceNorm(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  if (!CheckPrimitiveType(cnode, kPrimInstanceNorm)) {
    return false;
  }
  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  auto pre_node = cnode->input(1);
  if (!CheckPrimitiveType(pre_node, prim::kPrimConv2DFusion) && !CheckPrimitiveType(pre_node, prim::kPrimActivation)) {
    return true;
  }
  if (!utils::isa<CNode>(pre_node)) {
    return true;
  }
  std::vector<AnfNodePtr> pre_nodes;
  pre_nodes.push_back(pre_node);
  if (CheckPrimitiveType(pre_node, prim::kPrimActivation)) {
    pre_node = pre_node->cast<CNodePtr>()->input(1);
    if (!utils::isa<CNode>(pre_node) || !CheckPrimitiveType(pre_node, prim::kPrimConv2DFusion)) {
      return true;
    }
    pre_nodes.push_back(pre_node);
  }
  bool is_nc = false;
  for (const auto &node : pre_nodes) {
    auto node_users = manager->node_users()[node];
    is_nc = is_nc || std::any_of(node_users.begin(), node_users.end(), [](const std::pair<AnfNodePtr, int> &node_user) {
              return !CheckPrimitiveType(node_user.first, kPrimInstanceNorm);
            });
  }
  return is_nc;
}

int SpecialNodePostProcess::HandleInstanceNorm(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_CHECK_TRUE_RET(prim != nullptr, lite::RET_ERROR);
  if (prim->GetAttr(ops::kFormat) == nullptr) {
    MS_LOG(ERROR) << "The node should have format attribute.";
    return lite::RET_ERROR;
  }
  auto format = GetValue<int64_t>(prim->GetAttr(ops::kFormat));
  if (format == mindspore::NCHW) {
    return lite::RET_OK;
  }
  if (format != mindspore::NHWC) {
    MS_LOG(ERROR) << "format attribute is invalid.";
    return lite::RET_ERROR;
  }
  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  auto pre_transpose =
    GenTransposeNode(func_graph, cnode->input(1), kNH2NC, cnode->fullname_with_scope() + "_pre_nh2nc");
  MS_CHECK_TRUE_RET(pre_transpose != nullptr, lite::RET_ERROR);
  auto pre_trans_prim = GetValueNode<PrimitivePtr>(pre_transpose->input(0));
  MS_CHECK_TRUE_RET(pre_trans_prim != nullptr, lite::RET_ERROR);
  (void)pre_trans_prim->AddAttr(ops::kFormat, MakeValue<int64_t>(mindspore::NHWC));
  auto abstract = GetCNodeInputAbstract(cnode, 1);
  if (abstract != nullptr) {
    auto shape = GenerateNewShape(abstract);
    auto pre_trans_abstract = abstract->Clone();
    pre_trans_abstract->set_shape(std::make_shared<abstract::Shape>(shape));
    pre_transpose->set_abstract(pre_trans_abstract);
  }
  manager->SetEdge(cnode, 1, pre_transpose);
  auto post_transpose = GenTransposeNode(func_graph, cnode, kNC2NH, cnode->fullname_with_scope() + "_post_nc2nh");
  MS_CHECK_TRUE_RET(post_transpose != nullptr, lite::RET_ERROR);
  auto post_trans_prim = GetValueNode<PrimitivePtr>(post_transpose->input(0));
  MS_CHECK_TRUE_RET(post_trans_prim != nullptr, lite::RET_ERROR);
  (void)post_trans_prim->AddAttr(ops::kFormat, MakeValue<int64_t>(mindspore::NCHW));
  (void)prim->AddAttr(ops::kFormat, MakeValue<int64_t>(mindspore::NCHW));
  abstract = cnode->abstract();
  if (abstract != nullptr) {
    post_transpose->set_abstract(abstract->Clone());
    auto shape = GenerateNewShape(abstract);
    abstract->set_shape(std::make_shared<abstract::Shape>(shape));
  }
  (void)manager->Replace(cnode, post_transpose);
  return lite::RET_OK;
}
}  // namespace opt
}  // namespace mindspore
