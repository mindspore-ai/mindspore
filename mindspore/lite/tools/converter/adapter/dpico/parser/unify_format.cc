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

#include "parser/unify_format.h"
#include <set>
#include "common/check_base.h"
#include "common/format_utils.h"
#include "parser/parser_utils.h"
#include "ops/tuple_get_item.h"
#include "ops/adam.h"
#include "ops/sgd.h"
#include "ops/fusion/conv2d_fusion.h"
#include "ops/fusion/conv2d_transpose_fusion.h"
#include "ops/op_name.h"

namespace mindspore {
namespace lite {
void UnifyFormatToNHWC::GetTransNodeFormatType(const api::CNodePtr &cnode, dpico::TransTypePair *trans_info) {
  MS_ASSERT(cnode != nullptr && trans_info != nullptr);
  auto prim_node = cnode->input(0);
  auto prim = api::GetValueNode<api::PrimitivePtr>(prim_node);
  MS_ASSERT(prim != nullptr);
  auto &specify_ops = dpico::GetAssignedFormatOpSet();
  if (specify_ops.find(prim->name()) != specify_ops.end()) {
    trans_info->pre_ = dpico::kNCHW2NHWC;
    trans_info->post_ = dpico::kNHWC2NCHW;
  }
}

STATUS UnifyFormatToNHWC::GenNewInput(const api::FuncGraphPtr &func_graph, const api::CNodePtr &cnode,
                                      const std::vector<int> &perm, bool before, size_t index) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  api::AnfNodePtr trans_input = before ? cnode->input(index) : cnode->cast<api::AnfNodePtr>();
  std::string trans_name = before ? cnode->fullname_with_scope() + "_pre_" + std::to_string(index - 1)
                                  : cnode->fullname_with_scope() + "_post";
  auto trans_cnode = dpico::GenTransposeNode(func_graph, trans_input, perm, trans_name);
  if (trans_cnode == nullptr) {
    MS_LOG(ERROR) << "trans_cnode is nullptr.";
    return RET_ERROR;
  }
  auto abstract = trans_input->abstract();
  if (abstract != nullptr) {
    trans_cnode->set_abstract(abstract->Clone());
  }
  auto trans_prim = api::GetValueNode<api::PrimitivePtr>(trans_cnode->input(0));
  if (trans_prim == nullptr) {
    MS_LOG(ERROR) << "trans_prim is nullptr.";
    return RET_ERROR;
  }
  if (perm == dpico::kNC2NH) {
    (void)trans_prim->AddAttr(ops::kFormat, api::MakeValue<int64_t>(NCHW));
  } else if (perm == dpico::kNH2NC) {
    (void)trans_prim->AddAttr(ops::kFormat, api::MakeValue<int64_t>(NHWC));
  }
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    manager = api::FuncGraphManager::Manage(func_graph, true);
  }
  MS_CHECK_TRUE_MSG(manager != nullptr, RET_ERROR, "manager is nullptr");
  if (before) {
    manager->SetEdge(cnode, index, trans_cnode);
  } else {
    (void)manager->Replace(cnode, trans_cnode);
  }
  return lite::RET_OK;
}

STATUS UnifyFormatToNHWC::InsertPreTransNode(const api::FuncGraphPtr &func_graph, const api::CNodePtr &cnode,
                                             const std::vector<int> &perm) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  auto prim_node = cnode->input(0);
  auto prim = api::GetValueNode<api::PrimitivePtr>(prim_node);
  MS_ASSERT(prim != nullptr);
  auto &specify_ops = dpico::GetAssignedFormatOpSet();
  if (specify_ops.find(prim->name()) == specify_ops.end()) {
    MS_LOG(ERROR) << "p don't meet nhwc condition.";
    return lite::RET_ERROR;
  }
  if (GenNewInput(func_graph, cnode, perm, true, 1) != lite::RET_OK) {
    MS_LOG(ERROR) << "generate a transpose node failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

STATUS UnifyFormatToNHWC::InsertPostTransNode(const api::FuncGraphPtr &func_graph, const api::CNodePtr &cnode,
                                              const std::vector<int> &perm) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  if (!cnode->abstract()->isa<api::AbstractTuple>()) {
    if (GenNewInput(func_graph, cnode, perm, false) != lite::RET_OK) {
      MS_LOG(ERROR) << "generate a new input failed.";
      return lite::RET_ERROR;
    }
  } else {
    MS_CHECK_TRUE_MSG(func_graph->manager() != nullptr, RET_ERROR, "manager is nullptr");
    auto node_users = func_graph->manager()->GetUsers(cnode);
    for (auto &node_user : node_users) {
      auto post_node = node_user.first;
      if (post_node == nullptr) {
        MS_LOG(ERROR) << "post_node is nullptr.";
        return RET_ERROR;
      }
      if (!dpico::CheckPrimitiveType(post_node, api::MakeShared<ops::TupleGetItem>())) {
        MS_LOG(ERROR) << "post node is invalid.";
        return lite::RET_ERROR;
      }
      if (func_graph->manager()->GetUsers(post_node).empty()) {
        continue;
      }
      auto post_cnode = post_node->cast<api::CNodePtr>();
      if (GenNewInput(func_graph, post_cnode, perm, false) != lite::RET_OK) {
        MS_LOG(ERROR) << "generate a new input failed.";
        return lite::RET_ERROR;
      }
    }
  }
  return lite::RET_OK;
}

STATUS UnifyFormatToNHWC::HandleGraphInput(const api::FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto graph_input = func_graph->get_inputs();
  for (auto &input : graph_input) {
    if (input == nullptr) {
      MS_LOG(ERROR) << "input is nullptr.";
      return RET_ERROR;
    }
    auto input_param = input->cast<api::ParameterPtr>();
    MS_ASSERT(input_param != nullptr);
    auto abstract = input_param->abstract();
    MS_ASSERT(abstract != nullptr);
    ShapeVector shape;
    if (dpico::FetchShapeFromAbstract(abstract, &shape) != lite::RET_OK) {
      MS_LOG(ERROR) << "fetch shape failed." << input->fullname_with_scope();
      return lite::RET_ERROR;
    }
    if (shape.size() != dpico::kDims4) {
      continue;
    }
    ShapeVector transfer_shape = {shape[0], shape[dpico::kInputIndex2], shape[dpico::kInputIndex3], shape[1]};
    api::CNodePtr trans_cnode =
      dpico::GenTransposeNode(func_graph, input, dpico::kNH2NC, input->fullname_with_scope() + "_nh2nc");
    if (trans_cnode == nullptr) {
      MS_LOG(ERROR) << "create transpose cnode failed.";
      return lite::RET_ERROR;
    }
    auto trans_prim = api::GetValueNode<api::PrimitivePtr>(trans_cnode->input(0));
    MS_ASSERT(trans_prim != nullptr);
    (void)trans_prim->AddAttr(ops::kFormat, api::MakeValue<int64_t>(NHWC));
    trans_cnode->set_abstract(abstract->Clone());
    auto transfer_shape_ptr = api::MakeShared<api::Shape>(transfer_shape);
    if (transfer_shape_ptr == nullptr) {
      MS_LOG(ERROR) << "transfer_shape_ptr is nullptr.";
      return RET_ERROR;
    }
    abstract->set_shape(transfer_shape_ptr);
    MS_CHECK_TRUE_MSG(func_graph->manager() != nullptr, RET_ERROR, "manager is nullptr");
    if (!func_graph->manager()->Replace(input, trans_cnode)) {
      MS_LOG(ERROR) << "replace cnode failed.";
      return RET_ERROR;
    }
  }
  return lite::RET_OK;
}

STATUS UnifyFormatToNHWC::HandleGraphNode(const api::FuncGraphPtr &func_graph, const api::CNodePtr &cnode) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  dpico::TransTypePair trans_info;
  GetTransNodeFormatType(cnode, &trans_info);
  if (trans_info.pre_ == dpico::kNONE || trans_info.post_ == dpico::kNONE) {
    return lite::RET_NO_CHANGE;
  }
  auto before_perm = trans_info.pre_ == dpico::kNHWC2NCHW ? dpico::kNH2NC : dpico::kNC2NH;
  auto after_perm = trans_info.post_ == dpico::kNCHW2NHWC ? dpico::kNC2NH : dpico::kNH2NC;
  if (InsertPreTransNode(func_graph, cnode, before_perm) != lite::RET_OK) {
    MS_LOG(ERROR) << "insert pre node failed." << cnode->fullname_with_scope();
    return lite::RET_ERROR;
  }
  if (dpico::CheckPrimitiveType(cnode, api::MakeShared<ops::Adam>()) ||
      dpico::CheckPrimitiveType(cnode, api::MakeShared<ops::SGD>())) {
    return lite::RET_OK;
  }
  auto prim = api::GetValueNode<api::PrimitivePtr>(cnode->input(0));
  if (prim == nullptr) {
    MS_LOG(ERROR) << "current node's prim is nullptr, " << cnode->fullname_with_scope();
    return lite::RET_ERROR;
  }
  (void)prim->AddAttr(ops::kFormat, api::MakeValue<int64_t>(mindspore::NHWC));
  if (InsertPostTransNode(func_graph, cnode, after_perm) != lite::RET_OK) {
    MS_LOG(ERROR) << "insert post node failed." << cnode->fullname_with_scope();
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

bool UnifyFormatToNHWC::BasicProcess(const api::FuncGraphPtr &func_graph, bool main_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto manager = api::FuncGraphManager::Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr.";
    return false;
  }
  auto node_list = api::FuncGraph::TopoSort(func_graph->get_return());
  int status;
  for (auto &node : node_list) {
    if (!api::utils::isa<api::CNodePtr>(node)) {
      continue;
    }
    auto cnode = node->cast<api::CNodePtr>();
    if (dpico::IsSpecialType(cnode)) {
      continue;
    }
    status = HandleGraphNode(func_graph, cnode);
    if (status != lite::RET_OK && status != lite::RET_NO_CHANGE) {
      return false;
    }
  }
  if (main_graph) {
    if (HandleGraphInput(func_graph) != lite::RET_OK) {
      return false;
    }
  }
  return true;
}

STATUS UnifyFormatToNHWC::ConvWeightFormatTrans(const api::FuncGraphPtr &graph,
                                                std::set<api::AnfNodePtr> *has_visited) {
  MS_ASSERT(graph != nullptr && has_visited != nullptr);
  auto node_list = api::FuncGraph::TopoSort(graph->get_return());
  for (auto &node : node_list) {
    if (!api::utils::isa<api::CNodePtr>(node)) {
      continue;
    }
    auto cnode = node->cast<api::CNodePtr>();
    if (!dpico::CheckPrimitiveType(node, api::MakeShared<ops::Conv2DFusion>()) &&
        !dpico::CheckPrimitiveType(node, api::MakeShared<ops::Conv2dTransposeFusion>())) {
      continue;
    }
    if (has_visited->find(node) != has_visited->end()) {
      continue;
    }
    has_visited->insert(node);
    auto status = lite::UnifyConvWeightFormat(graph, cnode, mindspore::KCHW, mindspore::KHWC, has_visited);
    if (status != lite::RET_OK) {
      MS_LOG(ERROR) << "unify conv weight failed, current node name is " << cnode->fullname_with_scope();
      return status;
    }
  }
  return lite::RET_OK;
}

bool UnifyFormatToNHWC::Run(const api::FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto node_list = api::FuncGraph::TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    auto prim = api::GetValueNode<api::PrimitivePtr>(node);
    if (prim == nullptr) {
      continue;
    }
  }
  std::set<api::AnfNodePtr> has_visited;
  auto status = ConvWeightFormatTrans(func_graph, &has_visited);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "Conv2D weight FormatTrans failed: " << status;
    return false;
  }
  if (!BasicProcess(func_graph, true)) {
    MS_LOG(ERROR) << "run framework transpose unify failed.";
    return false;
  }
  return true;
}
}  // namespace lite
}  // namespace mindspore
