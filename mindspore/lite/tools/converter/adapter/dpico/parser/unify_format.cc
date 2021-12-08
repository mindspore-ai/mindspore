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
#include "common/format_utils.h"
#include "ops/op_utils.h"
#include "parser/parser_utils.h"

namespace mindspore {
namespace lite {
void UnifyFormatToNHWC::GetTransNodeFormatType(const CNodePtr &cnode, dpico::TransTypePair *trans_info) {
  MS_ASSERT(cnode != nullptr && trans_info != nullptr);
  auto prim_node = cnode->input(0);
  auto prim = GetValueNode<PrimitivePtr>(prim_node);
  MS_ASSERT(prim != nullptr);
  auto &specify_ops = dpico::GetAssignedFormatOpSet();
  if (specify_ops.find(prim->name()) != specify_ops.end()) {
    trans_info->pre_ = dpico::kNCHW2NHWC;
    trans_info->post_ = dpico::kNHWC2NCHW;
  }
}

STATUS UnifyFormatToNHWC::GenNewInput(const api::FuncGraphPtr &func_graph, const CNodePtr &cnode, std::vector<int> perm,
                                      bool before, size_t index) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  AnfNodePtr trans_input = before ? cnode->input(index) : cnode;
  std::string trans_name = before ? cnode->fullname_with_scope() + "_pre_" + std::to_string(index - 1)
                                  : cnode->fullname_with_scope() + "_post";
  auto trans_cnode = dpico::GenTransposeNode(func_graph, trans_input, perm, trans_name);
  auto abstract = trans_input->abstract();
  if (abstract != nullptr) {
    trans_cnode->set_abstract(abstract->Clone());
  }
  auto trans_prim = GetValueNode<PrimitivePtr>(trans_cnode->input(0));
  if (perm == dpico::kNC2NH) {
    trans_prim->AddAttr(ops::kFormat, MakeValue<int64_t>(NCHW));
  } else if (perm == dpico::kNH2NC) {
    trans_prim->AddAttr(ops::kFormat, MakeValue<int64_t>(NHWC));
  }
  auto manager = func_graph->get_manager();
  if (manager == nullptr) {
    manager = api::FuncGraphManager::Manage(func_graph, true);
  }
  MS_ASSERT(manager != nullptr);
  if (before) {
    manager->SetEdge(cnode, index, trans_cnode);
  } else {
    manager->Replace(cnode, trans_cnode);
  }
  return lite::RET_OK;
}

STATUS UnifyFormatToNHWC::InsertPreTransNode(const api::FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                             const std::vector<int> &perm) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  auto prim_node = cnode->input(0);
  auto prim = GetValueNode<PrimitivePtr>(prim_node);
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

STATUS UnifyFormatToNHWC::InsertPostTransNode(const api::FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                              const std::vector<int> &perm) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  if (!cnode->abstract()->isa<abstract::AbstractTuple>()) {
    if (GenNewInput(func_graph, cnode, perm, false) != lite::RET_OK) {
      MS_LOG(ERROR) << "generate a new input failed.";
      return lite::RET_ERROR;
    }
  } else {
    MS_ASSERT(func_graph->get_manager() != nullptr);
    auto node_map = func_graph->get_manager()->node_users();
    auto &node_users = node_map[cnode];
    for (auto &node_user : node_users) {
      auto post_node = node_user.first;
      if (!dpico::CheckPrimitiveType(post_node, prim::kPrimTupleGetItem)) {
        MS_LOG(ERROR) << "post node is invalid.";
        return lite::RET_ERROR;
      }
      if (node_map[post_node].empty()) {
        continue;
      }
      auto post_cnode = post_node->cast<CNodePtr>();
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
    auto input_param = input->cast<ParameterPtr>();
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
    CNodePtr trans_cnode =
      dpico::GenTransposeNode(func_graph, input, dpico::kNH2NC, input->fullname_with_scope() + "_nh2nc");
    if (trans_cnode == nullptr) {
      MS_LOG(ERROR) << "create transpose cnode failed.";
      return lite::RET_ERROR;
    }
    auto trans_prim = GetValueNode<PrimitivePtr>(trans_cnode->input(0));
    MS_ASSERT(trans_prim != nullptr);
    trans_prim->AddAttr(ops::kFormat, MakeValue<int64_t>(NHWC));
    trans_cnode->set_abstract(abstract->Clone());
    auto transfer_shape_ptr = std::make_shared<abstract::Shape>(transfer_shape);
    if (transfer_shape_ptr == nullptr) {
      MS_LOG(ERROR) << "transfer_shape_ptr is nullptr.";
      return RET_ERROR;
    }
    abstract->set_shape(transfer_shape_ptr);
    MS_ASSERT(func_graph->get_manager() != nullptr);
    func_graph->get_manager()->Replace(input, trans_cnode);
  }
  return lite::RET_OK;
}

STATUS UnifyFormatToNHWC::HandleGraphNode(const api::FuncGraphPtr &func_graph, const CNodePtr &cnode) {
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
  if (dpico::CheckPrimitiveType(cnode, prim::kPrimAdam) || dpico::CheckPrimitiveType(cnode, prim::kPrimSGD)) {
    return lite::RET_OK;
  }
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  if (prim == nullptr) {
    MS_LOG(ERROR) << "current node's prim is nullptr, " << cnode->fullname_with_scope();
    return lite::RET_ERROR;
  }
  prim->AddAttr(ops::kFormat, MakeValue<int64_t>(mindspore::NHWC));
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
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (dpico::IsSpecialType(cnode)) {
      continue;
    }
    if (dpico::CheckPrimitiveType(node, prim::kPrimIf) || dpico::CheckPrimitiveType(node, prim::kPrimWhile)) {
      auto sub_func_graph = api::FuncGraph::GetFuncGraphFromAnfNode(cnode->input(1));
      if (sub_func_graph == nullptr) {
        MS_LOG(ERROR) << "sub graph is nullptr.";
        return false;
      }
      (void)BasicProcess(sub_func_graph, false);
      sub_func_graph = api::FuncGraph::GetFuncGraphFromAnfNode(cnode->input(dpico::kInputIndex2));
      if (sub_func_graph == nullptr) {
        MS_LOG(ERROR) << "sub graph is nullptr.";
        return false;
      }
      (void)BasicProcess(sub_func_graph, false);
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

STATUS UnifyFormatToNHWC::ConvWeightFormatTrans(const api::FuncGraphPtr &graph, std::set<AnfNodePtr> *has_visited) {
  MS_ASSERT(graph != nullptr && has_visited != nullptr);
  auto node_list = api::FuncGraph::TopoSort(graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (dpico::CheckPrimitiveType(node, prim::kPrimIf) || dpico::CheckPrimitiveType(node, prim::kPrimWhile)) {
      auto sub_func_graph = api::FuncGraph::GetFuncGraphFromAnfNode(cnode->input(1));
      if (sub_func_graph == nullptr) {
        MS_LOG(ERROR) << "subgraph is nullptr.";
        return false;
      }
      if (ConvWeightFormatTrans(sub_func_graph, has_visited) != lite::RET_OK) {
        MS_LOG(ERROR) << "transform conv weight format failed.";
        return lite::RET_ERROR;
      }
      sub_func_graph = api::FuncGraph::GetFuncGraphFromAnfNode(cnode->input(dpico::kInputIndex2));
      if (sub_func_graph == nullptr) {
        MS_LOG(ERROR) << "subgraph is nullptr.";
        return false;
      }
      if (ConvWeightFormatTrans(sub_func_graph, has_visited) != lite::RET_OK) {
        MS_LOG(ERROR) << "transform conv weight format failed.";
        return lite::RET_ERROR;
      }
      continue;
    }
    if (!dpico::CheckPrimitiveType(node, prim::kPrimConv2DFusion) &&
        !dpico::CheckPrimitiveType(node, prim::kPrimConv2dTransposeFusion)) {
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
    auto prim = GetValueNode<PrimitivePtr>(node);
    if (prim == nullptr) {
      continue;
    }
  }
  std::set<AnfNodePtr> has_visited;
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
