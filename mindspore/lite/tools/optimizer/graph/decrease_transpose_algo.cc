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

#include "tools/optimizer/graph/decrease_transpose_algo.h"
#include <queue>
#include <set>
#include <unordered_map>
#include <utility>
#include "ops/op_utils.h"
#include "src/common/common.h"
#include "src/common/utils.h"
#include "tools/common/tensor_util.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace opt {
namespace {
STATUS FindAreaSurroundedByTranspose(const FuncGraphPtr &func_graph, const CNodePtr &root_node,
                                     std::set<CNodePtr> *in_nodes, std::set<CNodePtr> *out_nodes,
                                     std::set<CNodePtr> *middle_nodes) {
  MS_ASSERT(func_graph != nullptr && root_node != nullptr);
  MS_ASSERT(in_nodes != nullptr && out_nodes != nullptr && middle_nodes != nullptr);
  std::queue<CNodePtr> queue_nodes{};
  queue_nodes.push(root_node);
  std::queue<bool> is_pre_nodes;
  is_pre_nodes.push(true);
  while (!queue_nodes.empty()) {
    auto cur_node = queue_nodes.front();
    auto is_pre_node = is_pre_nodes.front();
    queue_nodes.pop();
    is_pre_nodes.pop();
    if (CheckPrimitiveType(cur_node, prim::kPrimTranspose)) {
      if (is_pre_node) {
        in_nodes->insert(cur_node);
      } else {
        out_nodes->insert(cur_node);
        continue;
      }
    }
    if (middle_nodes->find(cur_node) != middle_nodes->end()) {
      continue;
    }
    if (in_nodes->find(cur_node) == in_nodes->end()) {
      middle_nodes->insert(cur_node);
      // insert pre nodes.
      auto origin_inputs = cur_node->inputs();
      lite::RemoveIfDepend(cur_node);
      for (size_t i = 1; i < cur_node->size(); ++i) {
        if (!utils::isa<CNodePtr>(cur_node->input(i))) {
          continue;
        }
        auto cur_node_input = cur_node->input(i)->cast<CNodePtr>();
        MS_ASSERT(cur_node_input != nullptr);
        if (middle_nodes->find(cur_node_input) != middle_nodes->end() ||
            in_nodes->find(cur_node_input) != in_nodes->end()) {
          continue;
        }
        queue_nodes.push(cur_node_input);
        is_pre_nodes.push(true);
      }
      if (CheckIsAllInputsParam(cur_node)) {
        in_nodes->insert(cur_node);
      }
      cur_node->set_inputs(origin_inputs);
    }
    // insert post nodes
    auto cur_node_users = func_graph->manager()->node_users()[cur_node];
    for (auto &cur_node_user : cur_node_users) {
      if (!utils::isa<CNodePtr>(cur_node_user.first)) {
        MS_LOG(ERROR) << "post node is not cnode.";
        return lite::RET_ERROR;
      }
      auto cur_node_post = cur_node_user.first->cast<CNodePtr>();
      MS_CHECK_TRUE_MSG(cur_node_post != nullptr, RET_ERROR, "cast ptr failed");
      if (middle_nodes->find(cur_node_post) != middle_nodes->end() ||
          out_nodes->find(cur_node_post) != out_nodes->end()) {
        continue;
      }
      queue_nodes.push(cur_node_post);
      is_pre_nodes.push(false);
    }
    if (cur_node_users.empty()) {
      out_nodes->insert(cur_node);
    }
  }
  return lite::RET_OK;
}

void SetTransType(const std::set<CNodePtr> &cnodes, FormatTransNodeType *trans_type) {
  MS_ASSERT(trans_type != nullptr);
  FormatTransNodeType local_trans_type;
  for (auto &cnode : cnodes) {
    std::vector<int> perm;
    if (!CheckPrimitiveType(cnode, prim::kPrimTranspose) || GetTransposePerm(cnode, &perm) != lite::RET_OK ||
        (perm != kNH2NC && perm != kNC2NH)) {
      *trans_type = kNONE;
      return;
    }
    local_trans_type = perm == kNH2NC ? kNHWC2NCHW : kNCHW2NHWC;
    *trans_type = *trans_type == kNONE ? local_trans_type : *trans_type;
    if (*trans_type != local_trans_type) {
      *trans_type = kNONE;
      return;
    }
  }
}

bool JudgeCanOptimizerForMultiOp(const std::set<CNodePtr> &in_nodes, const std::set<CNodePtr> &out_nodes,
                                 const std::set<CNodePtr> &middle_nodes, TransTypePair *trans_info) {
  MS_ASSERT(trans_info != nullptr);
  SetTransType(in_nodes, &trans_info->pre_);
  if (trans_info->pre_ == kNONE) {
    return false;
  }
  SetTransType(out_nodes, &trans_info->post_);
  if (trans_info->post_ == kNONE) {
    return false;
  }
  if (trans_info->pre_ == trans_info->post_) {
    return false;
  }
  TransposeStrategy transpose_strategy;
  for (auto &middle_cnode : middle_nodes) {
    if (IsSpecialType(middle_cnode)) {
      continue;
    }
    auto middle_node_prim = GetValueNode<PrimitivePtr>(middle_cnode->input(0));
    MS_CHECK_TRUE_MSG(middle_node_prim != nullptr, false, "GetValueNode failed");
    if (!transpose_strategy.CanChangeOpAxis(middle_cnode)) {
      return false;
    }
  }
  return true;
}

int ConvertTensorToNCOrNH(const FuncGraphPtr &func_graph, const CNodePtr &cnode, size_t index, FmkType fmk_type,
                          bool train_flag, FormatTransNodeType trans_type) {
  MS_ASSERT(cnode != nullptr);
  if (utils::isa<CNodePtr>(cnode->input(index))) {
    return lite::RET_OK;
  }
  lite::DataInfo data_info;
  int status = 0;
  if (utils::isa<ParameterPtr>(cnode->input(index))) {
    auto input_node = cnode->input(index)->cast<ParameterPtr>();
    MS_CHECK_TRUE_MSG(input_node != nullptr, lite::RET_ERROR, "input_node is nullptr");
    if (!input_node->has_default()) {
      return lite::RET_OK;
    }
    status = lite::FetchDataFromParameterNode(cnode, index, fmk_type, train_flag, &data_info);
  } else {
    status = lite::FetchDataFromValueNode(cnode, index, fmk_type, train_flag, &data_info);
  }
  if (status != lite::RET_OK) {
    return lite::RET_ERROR;
  }
  if (data_info.shape_.empty() ||
      (data_info.data_type_ != kNumberTypeFloat32 && data_info.data_type_ != kNumberTypeFloat)) {
    return lite::RET_OK;
  }
  ShapeVector expand_shape(data_info.shape_.begin(), data_info.shape_.end());
  if (data_info.shape_.size() == 1) {
    expand_shape = {1, 1, 1, data_info.shape_[0]};
  } else if (data_info.shape_.size() == kInputSizeTwo) {
    expand_shape = {1, 1, data_info.shape_[0], data_info.shape_[1]};
  } else if (data_info.shape_.size() == kInputSizeThree) {
    expand_shape = {1, data_info.shape_[0], data_info.shape_[1], data_info.shape_[kInputIndexTwo]};
  }
  auto tensor = std::make_shared<tensor::Tensor>(static_cast<TypeId>(data_info.data_type_), expand_shape,
                                                 data_info.data_.data(), data_info.data_.size());
  MS_CHECK_TRUE_MSG(tensor != nullptr, lite::RET_ERROR, "tensor is nullptr");
  if (trans_type == kNHWC2NCHW) {
    (void)TransFilterFormat(tensor, schema::Format_KHWC, schema::Format_KCHW);
  } else {
    (void)TransFilterFormat(tensor, schema::Format_KCHW, schema::Format_KHWC);
  }
  auto param_node = func_graph->add_parameter();
  MS_CHECK_TRUE_MSG(param_node != nullptr, lite::RET_ERROR, "add_parameter failed");
  param_node->set_name(cnode->input(index)->fullname_with_scope());
  status = lite::InitParameterFromTensorInfo(param_node, tensor);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "init parameter from tensor info failed";
    return lite::RET_ERROR;
  }
  auto tr = func_graph->manager()->Transact();
  tr.SetEdge(cnode, index, param_node);
  tr.Commit();
  return lite::RET_OK;
}
}  // namespace

STATUS DecreaseTransposeAlgo::PostTransposeFusion(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  if (!CheckPrimitiveType(cnode, prim::kPrimTranspose)) {
    return lite::RET_OK;
  }
  std::vector<int> cur_perm;
  if (GetTransposePerm(cnode, &cur_perm) != lite::RET_OK) {
    MS_LOG(ERROR) << "get transpose perm failed.";
    return lite::RET_ERROR;
  }
  auto node_users = func_graph->manager()->node_users()[cnode];
  for (auto &node_user : node_users) {
    auto post_node = node_user.first;
    if (CheckPrimitiveType(post_node, prim::kPrimTranspose)) {
      std::vector<int> post_trans_perm;
      auto post_trans_node = post_node->cast<CNodePtr>();
      MS_ASSERT(post_trans_node != nullptr);
      if (GetTransposePerm(post_trans_node, &post_trans_perm) != lite::RET_OK) {
        MS_LOG(ERROR) << "get post transpose node perm failed.";
        return lite::RET_ERROR;
      }
      if ((cur_perm == kNH2NC && post_trans_perm == kNC2NH) || (cur_perm == kNC2NH && post_trans_perm == kNH2NC)) {
        func_graph->manager()->Replace(post_node, cnode->input(1));
      }
    }
  }
  return lite::RET_OK;
}

STATUS DecreaseTransposeAlgo::GenNewInput(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                          const std::vector<int> perm, bool before, size_t index) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  AnfNodePtr new_input = nullptr;
  new_input = transpose_strategy_.TransposePairFuseWhenInsert(func_graph, cnode, perm, before, index);
  if (new_input == nullptr) {
    MS_LOG(ERROR) << "generate a transpose node failed.";
    return lite::RET_ERROR;
  }
  if (new_input == cnode->input(index) || new_input == cnode) {
    return lite::RET_OK;
  } else if (utils::isa<CNodePtr>(new_input)) {
    auto new_cnode_input = new_input->cast<CNodePtr>();
    MS_ASSERT(new_cnode_input != nullptr);
    int status = lite::RET_OK;
    if (CheckPrimitiveType(new_cnode_input, prim::kPrimTranspose)) {
      status = node_infer_shape_.InferShape(new_cnode_input);
    }
    if (status != lite::RET_OK && status != lite::RET_INFER_INVALID) {
      MS_LOG(ERROR) << "infer shape failed.";
      return lite::RET_ERROR;
    }
  }
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    manager = Manage(func_graph, true);
  }
  auto tr = manager->Transact();
  if (before) {
    tr.SetEdge(cnode, index, new_input);
    tr.Commit();
  } else {
    func_graph->manager()->Replace(cnode, new_input);
    if (PostTransposeFusion(func_graph, new_input->cast<CNodePtr>()) != lite::RET_OK) {
      MS_LOG(ERROR) << "post transpose fusion failed.";
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}

STATUS DecreaseTransposeAlgo::InsertPreTransNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                                 const std::vector<int> &perm) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  auto prim_node = cnode->input(0);
  MS_CHECK_TRUE_MSG(prim_node != nullptr, lite::RET_ERROR, "prim_node is nullptr");
  auto prim = GetValueNode<PrimitivePtr>(prim_node);
  MS_CHECK_TRUE_MSG(prim != nullptr, lite::RET_ERROR, "GetValueNode Failed");
  auto &specify_nhwc_op_map = GetNHWCOpMap();
  auto &specify_nchw_op_map = GetNCHWOpMap();
  if (specify_nhwc_op_map.find(prim->name()) == specify_nhwc_op_map.end() &&
      specify_nchw_op_map.find(prim->name()) == specify_nchw_op_map.end()) {
    MS_LOG(ERROR) << "op don't meet nhwc condition.";
    return lite::RET_ERROR;
  }
  std::vector<size_t> insert_index = specify_nchw_op_map.find(prim->name()) == specify_nchw_op_map.end()
                                       ? specify_nhwc_op_map.at(prim->name())
                                       : specify_nchw_op_map.at(prim->name());
  if (insert_index.empty()) {
    if (CheckPrimitiveType(cnode, prim::kPrimResizeGrad) && prim->GetAttr(ops::kMethod) != nullptr &&
        GetValue<int64_t>(prim->GetAttr(ops::kMethod)) == static_cast<int64_t>(mindspore::ResizeMethod::NEAREST)) {
      insert_index.push_back(1);
    } else {
      for (size_t i = 1; i < cnode->size(); ++i) {
        insert_index.push_back(i);
      }
    }
  }
  for (auto &index : insert_index) {
    if (GenNewInput(func_graph, cnode, perm, true, index) != lite::RET_OK) {
      MS_LOG(ERROR) << "generate a new input failed.";
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}

STATUS DecreaseTransposeAlgo::InsertPreTransNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                                 TransTypePair *trans_insert_info) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  MS_ASSERT(trans_insert_info != nullptr);
  TransTypePair trans_info;
  auto origin_inputs = cnode->inputs();
  lite::RemoveIfMakeTuple(cnode);
  RemoveIfMonad(cnode);
  if (!transpose_strategy_.CanFusionIfInsert(func_graph, cnode, &trans_info, trans_insert_info)) {
    cnode->set_inputs(origin_inputs);
    return lite::RET_NO_CHANGE;
  }
  cnode->set_inputs(origin_inputs);
  auto status = transpose_strategy_.ChangeOpAxis(func_graph, cnode, trans_insert_info->pre_);
  if (status == lite::RET_NOT_SUPPORT) {
    return lite::RET_NO_CHANGE;
  } else if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "change op attr failed.";
    return lite::RET_ERROR;
  }
  auto before_perm = trans_insert_info->pre_ == kNHWC2NCHW ? kNH2NC : kNC2NH;
  for (size_t i = 1; i < cnode->size(); ++i) {
    if (IsMonadNode(cnode->input(i))) {
      continue;
    }
    if (CheckPrimitiveType(cnode->input(i), prim::kPrimMakeTuple) ||
        CheckPrimitiveType(cnode->input(i), kPrimMakeTupleV2)) {
      auto input_make_tuple = cnode->input(i)->cast<CNodePtr>();
      MS_ASSERT(input_make_tuple != nullptr);
      for (size_t j = 1; j < input_make_tuple->size(); ++j) {
        if (GenNewInput(func_graph, input_make_tuple, before_perm, true, j) != lite::RET_OK) {
          MS_LOG(ERROR) << "generate a new input failed.";
          return lite::RET_ERROR;
        }
      }
      continue;
    }
    if (GenNewInput(func_graph, cnode, before_perm, true, i) != lite::RET_OK) {
      MS_LOG(ERROR) << "generate a new input failed.";
      return lite::RET_ERROR;
    }
  }
  status = ModifyCNodeFormat(cnode, trans_insert_info->pre_);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "ModifyCNodeFormat failed.";
    return lite::RET_ERROR;
  }
  status = node_infer_shape_.InferShape(cnode);
  if (status != lite::RET_OK && status != lite::RET_INFER_INVALID) {
    MS_LOG(ERROR) << "infer shape failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

STATUS DecreaseTransposeAlgo::InsertPostTransNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                                  const std::vector<int> &perm) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  if (!cnode->abstract()->isa<abstract::AbstractTuple>()) {
    if (GenNewInput(func_graph, cnode, perm, false) != lite::RET_OK) {
      MS_LOG(ERROR) << "generate a new input failed.";
      return lite::RET_ERROR;
    }
  } else {
    auto node_users = func_graph->manager()->node_users()[cnode];
    for (auto &node_user : node_users) {
      auto post_node = node_user.first;
      CNodePtr tuple_get_item = nullptr;
      if (!CheckPrimitiveType(post_node, prim::kPrimTupleGetItem)) {
        if (!train_flag_) {
          MS_LOG(ERROR) << "post node is invalid.";
          return lite::RET_ERROR;
        } else {
          tuple_get_item = GenTupleGetItemNode(func_graph, cnode, 0);
          MS_CHECK_TRUE_RET(tuple_get_item != nullptr, lite::RET_ERROR);
          post_node = tuple_get_item;
          func_graph->manager()->Replace(cnode, tuple_get_item);
        }
      }
      if (func_graph->manager()->node_users()[post_node].empty()) {
        continue;
      }
      auto post_cnode = post_node->cast<CNodePtr>();
      MS_ASSERT(post_cnode != nullptr);
      if (GenNewInput(func_graph, post_cnode, perm, false) != lite::RET_OK) {
        MS_LOG(ERROR) << "generate a new input failed.";
        return lite::RET_ERROR;
      }
      if (tuple_get_item != nullptr) {
        func_graph->manager()->Replace(tuple_get_item, tuple_get_item->input(1));
      }
    }
  }
  return lite::RET_OK;
}

STATUS DecreaseTransposeAlgo::HandleGraphMultiNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                                   std::set<CNodePtr> *visit_transposes) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr && visit_transposes != nullptr);
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_MSG(manager != nullptr, lite::RET_ERROR, "manager is nullptr");
  std::set<CNodePtr> middle_nodes{};
  std::set<CNodePtr> in_nodes{};
  std::set<CNodePtr> out_nodes{};
  auto status = FindAreaSurroundedByTranspose(func_graph, cnode, &in_nodes, &out_nodes, &middle_nodes);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "find an area surrounded by transpose failed.";
    return status;
  }
  for (auto &in_cnode : in_nodes) {
    if (CheckPrimitiveType(in_cnode, prim::kPrimTranspose)) {
      visit_transposes->insert(in_cnode);
    }
  }
  TransTypePair trans_info;
  if (!JudgeCanOptimizerForMultiOp(in_nodes, out_nodes, middle_nodes, &trans_info)) {
    return lite::RET_NO_CHANGE;
  }
  auto node_list = TopoSort(func_graph->get_return());
  std::vector<CNodePtr> middle_ops_vec;
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    if (middle_nodes.find(node->cast<CNodePtr>()) != middle_nodes.end()) {
      middle_ops_vec.push_back(node->cast<CNodePtr>());
      middle_nodes.erase(node->cast<CNodePtr>());
    }
  }
  for (auto &in_cnode : in_nodes) {
    manager->Replace(in_cnode, in_cnode->input(1));
  }
  for (auto &out_cnode : out_nodes) {
    manager->Replace(out_cnode, out_cnode->input(1));
  }
  for (auto &middle_cnode : middle_ops_vec) {
    if (IsSpecialType(middle_cnode)) {
      continue;
    }
    for (size_t i = 1; i < middle_cnode->size(); ++i) {
      status = ConvertTensorToNCOrNH(func_graph, middle_cnode, i, fmk_type_, train_flag_, trans_info.post_);
      if (status != lite::RET_OK) {
        MS_LOG(ERROR) << "ConvertTensorToNCOrNH failed.";
        return lite::RET_ERROR;
      }
    }
    status = transpose_strategy_.ChangeOpAxis(func_graph, middle_cnode, trans_info.post_);
    if (status != lite::RET_OK) {
      MS_LOG(ERROR) << "change op attr failed.";
      return lite::RET_ERROR;
    }
    status = ModifyCNodeFormat(middle_cnode, trans_info.post_);
    if (status != lite::RET_OK) {
      MS_LOG(ERROR) << "ModifyCNodeFormat failed.";
      return lite::RET_ERROR;
    }
    status = node_infer_shape_.InferShape(middle_cnode);
    if (status != lite::RET_OK && status != lite::RET_INFER_INVALID) {
      MS_LOG(ERROR) << "infer shape failed.";
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}

int DecreaseTransposeAlgo::SetSubGraphInput(const CNodePtr &cnode, const FuncGraphPtr &sub_graph) {
  MS_ASSERT(cnode != nullptr && sub_graph != nullptr);
  auto sub_inputs = sub_graph->get_inputs();
  sub_inputs_map_[sub_graph] = sub_inputs;
  for (auto &node : sub_inputs) {
    auto param_node = node->cast<ParameterPtr>();
    MS_ASSERT(param_node != nullptr);
    auto node_name = node->fullname_with_scope();
    auto last_underline = node_name.find_last_of("_");
    node_name = node_name.substr(0, last_underline);
    last_underline = node_name.find_last_of("_");
    auto index = 0;
    try {
      index = std::stoi(node_name.substr(last_underline + 1)) + static_cast<int>(kInputSizeThree);
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "Get index failed: " << e.what();
      return lite::RET_ERROR;
    }
    param_node->set_abstract(GetCNodeInputAbstract(cnode, index)->Clone());
    if (utils::isa<CNodePtr>(cnode->input(index))) {
      ShapeVector shape_vec = {-1};
      auto out_cnode = cnode->input(index)->cast<CNodePtr>();
      MS_ASSERT(out_cnode != nullptr);
      MS_ASSERT(trans_cnode != nullptr);
      auto out_prim = GetValueNode<PrimitivePtr>(out_cnode->input(0));
      MS_CHECK_TRUE_MSG(out_prim != nullptr, lite::RET_ERROR, "GetValueNode failed");
      if (out_prim->GetAttr(kInferDone) == nullptr || !GetValue<bool>(out_prim->GetAttr(kInferDone))) {
        param_node->abstract()->set_shape(std::make_shared<abstract::Shape>(shape_vec));
      }
    } else {
      lite::DataInfo data_info;
      if (utils::isa<ParameterPtr>(cnode->input(index))) {
        if (cnode->input(index)->cast<ParameterPtr>()->has_default()) {
          param_node->set_default_param(cnode->input(index)->cast<ParameterPtr>()->default_param());
        }
        continue;
      }
      auto status = lite::FetchDataFromValueNode(cnode, index, fmk_type_, train_flag_, &data_info);
      if (status != lite::RET_OK) {
        continue;
      }
      ShapeVector shape_vec(data_info.shape_.begin(), data_info.shape_.end());
      if (data_info.data_.empty()) {
        param_node->set_default_param(std::make_shared<tensor::Tensor>((TypeId)data_info.data_type_, shape_vec));
      } else {
        param_node->set_default_param(std::make_shared<tensor::Tensor>((TypeId)data_info.data_type_, shape_vec,
                                                                       data_info.data_.data(), data_info.data_.size()));
      }
    }
  }
  return lite::RET_OK;
}

int DecreaseTransposeAlgo::ResetSubGraphInput() {
  for (auto iter = sub_inputs_map_.begin(); iter != sub_inputs_map_.end(); ++iter) {
    auto &sub_graph = iter->first;
    auto &sub_inputs = iter->second;
    auto manager = sub_graph->manager();
    MS_ASSERT(manager != nullptr);
    for (auto &sub_input : sub_inputs) {
      auto param_node = sub_graph->add_parameter();
      MS_CHECK_TRUE_MSG(param_node != nullptr, lite::RET_ERROR, "add parameter failed");
      param_node->set_abstract(sub_input->abstract()->Clone());
      param_node->set_name(sub_input->fullname_with_scope());
      manager->Replace(sub_input, param_node);
      auto sub_param_input = sub_input->cast<ParameterPtr>();
      MS_ASSERT(sub_param_input != nullptr);
      sub_param_input->set_default_param(nullptr);
    }
  }
  return lite::RET_OK;
}

int DecreaseTransposeAlgo::SetSubGraphOutput(const FuncGraphPtr &sub_graph) {
  MS_ASSERT(sub_graph != nullptr);
  auto return_node = sub_graph->get_return();
  MS_ASSERT(return_node != nullptr);
  auto origin_input = return_node->inputs();
  lite::RemoveIfDepend(return_node);
  lite::RemoveIfMakeTuple(return_node);
  for (size_t i = 1; i < return_node->size(); ++i) {
    if (!CheckPrimitiveType(return_node->input(i), prim::kPrimTranspose)) {
      continue;
    }
    auto node_name = return_node->input(i)->fullname_with_scope();
    if (node_name.size() < kInputSizeFive || node_name.substr(node_name.size() - kInputSizeFive) != "_post") {
      continue;
    }
    auto trans_cnode = return_node->input(i)->cast<CNodePtr>();
    MS_ASSERT(trans_cnode != nullptr);
    auto trans_input = trans_cnode->input(1);
    auto trans_input_name = trans_input->fullname_with_scope();
    if (utils::isa<ParameterPtr>(trans_input)) {
      trans_input->cast<ParameterPtr>()->set_name(node_name);
    } else if (utils::isa<CNodePtr>(trans_input)) {
      trans_input->cast<CNodePtr>()->set_fullname_with_scope(node_name);
    }
    trans_input_name = trans_input_name.substr(0, trans_input_name.find_last_of("_")) + "_cnode";
    trans_cnode->set_fullname_with_scope(trans_input_name);
  }
  return_node->set_inputs(origin_input);
  return lite::RET_OK;
}

int DecreaseTransposeAlgo::SetSubGraphAbstract(const CNodePtr &cnode, const FuncGraphPtr &sub_graph) {
  MS_ASSERT(cnode != nullptr && sub_graph != nullptr);
  auto return_node = sub_graph->get_return();
  MS_CHECK_TRUE_MSG(return_node != nullptr, lite::RET_ERROR, "return_node is nullptr");
  auto origin_inputs = return_node->inputs();
  lite::RemoveIfDepend(return_node);
  lite::RemoveIfMakeTuple(return_node);
  AbstractBasePtrList abstract_list;
  bool infer_done = true;
  for (size_t i = 1; i < return_node->size(); ++i) {
    auto abstract_base = GetCNodeInputAbstract(return_node, i);
    MS_CHECK_TRUE_MSG(abstract_base != nullptr, lite::RET_ERROR, "GetCNodeInputAbstract failed");
    abstract_list.emplace_back(abstract_base->Clone());
    auto abstract_tensor = abstract_base->cast<abstract::AbstractTensorPtr>();
    MS_ASSERT(abstract_tensor != nullptr);
    auto shape_ptr = utils::cast<abstract::ShapePtr>(abstract_tensor->BuildShape());
    MS_ASSERT(shape_ptr != nullptr);
    auto shape = shape_ptr->shape();
    if (std::find(shape.begin(), shape.end(), -1) != shape.end()) {
      infer_done = false;
    }
    if (utils::isa<CNodePtr>(return_node->input(i))) {
      auto input_cnode = return_node->input(i)->cast<CNodePtr>();
      MS_CHECK_TRUE_MSG(input_cnode != nullptr, lite::RET_ERROR, "input_cnode is nullptr");
      if (CheckPrimitiveType(input_cnode, prim::kPrimTupleGetItem)) {
        input_cnode = input_cnode->input(1)->cast<CNodePtr>();
      }
      auto input_prim = GetValueNode<PrimitivePtr>(input_cnode->input(0));
      MS_CHECK_TRUE_MSG(input_prim != nullptr, lite::RET_ERROR, "GetValueNode failed");
      if (input_prim->GetAttr(kInferDone) == nullptr || !GetValue<bool>(input_prim->GetAttr(kInferDone))) {
        infer_done = false;
      }
    }
  }
  return_node->set_inputs(origin_inputs);
  if (utils::isa<abstract::AbstractTuplePtr>(cnode->abstract())) {
    cnode->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
  } else {
    if (abstract_list.size() != 1) {
      MS_LOG(ERROR) << "cnode output is invalid.";
    }
    cnode->set_abstract(abstract_list.front());
  }
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_CHECK_TRUE_MSG(prim != nullptr, lite::RET_ERROR, "GetValueNode Failed");
  prim->AddAttr(kInferDone, MakeValue<bool>(infer_done));

  return lite::RET_OK;
}

int DecreaseTransposeAlgo::ModifyCNodeFormat(const CNodePtr &cnode, FormatTransNodeType pre_trans_type) {
  MS_ASSERT(cnode != nullptr);
  if (pre_trans_type == kNONE) {
    return lite::RET_OK;
  }
  auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_CHECK_TRUE_MSG(primitive != nullptr, lite::RET_ERROR, "GetValueNode Failed");
  if (pre_trans_type == kNHWC2NCHW) {
    primitive->AddAttr(ops::kFormat, MakeValue<int64_t>(mindspore::NCHW));
  } else {
    primitive->AddAttr(ops::kFormat, MakeValue<int64_t>(mindspore::NHWC));
  }
  return lite::RET_OK;
}

bool DecreaseTransposeAlgo::DecreaseTransposeForSingleOp(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto manager = Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr.";
    return false;
  }
  auto node_list = TopoSort(func_graph->get_return());
  int status = 0;
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_ASSERT(cnode != nullptr);
    if (IsSpecialType(cnode)) {
      continue;
    }
    if (CheckPrimitiveType(node, prim::kPrimIf) || CheckPrimitiveType(node, prim::kPrimWhile)) {
      auto sub_func_graph = GetValueNode<FuncGraphPtr>(cnode->input(1));
      if (sub_func_graph == nullptr) {
        lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
        return false;
      }
      auto ret = SetSubGraphInput(cnode, sub_func_graph);
      if (ret != lite::RET_OK) {
        MS_LOG(ERROR) << "SetSubGraphInput failed";
        return false;
      }
      (void)DecreaseTransposeForSingleOp(sub_func_graph);
      ret = SetSubGraphOutput(sub_func_graph);
      if (ret != lite::RET_OK) {
        MS_LOG(ERROR) << "SetSubGraphOutput failed";
        return false;
      }
      sub_func_graph = GetValueNode<FuncGraphPtr>(cnode->input(kInputIndexTwo));
      if (sub_func_graph == nullptr) {
        lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
        return false;
      }
      ret = SetSubGraphInput(cnode, sub_func_graph);
      if (ret != lite::RET_OK) {
        MS_LOG(ERROR) << "SetSubGraphInput failed";
        return false;
      }
      (void)DecreaseTransposeForSingleOp(sub_func_graph);
      ret = SetSubGraphOutput(sub_func_graph);
      if (ret != lite::RET_OK) {
        MS_LOG(ERROR) << "SetSubGraphOutput failed";
        return false;
      }
      ret = SetSubGraphAbstract(cnode, sub_func_graph);
      if (ret != lite::RET_OK) {
        MS_LOG(ERROR) << "SetSubGraphAbstract failed";
        return false;
      }
      continue;
    }
    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    MS_CHECK_TRUE_MSG(prim != nullptr, false, "GetValueNode Failed");
    if (!IsDynamicFormatOp(prim->name())) {
      continue;
    }
    TransTypePair trans_insert_info;
    status = InsertPreTransNode(func_graph, cnode, &trans_insert_info);
    if (status == lite::RET_NO_CHANGE) {
      continue;
    } else if (status != lite::RET_OK) {
      MS_LOG(ERROR) << "insert pre node failed.";
      return false;
    }
    auto after_perm = trans_insert_info.post_ == kNHWC2NCHW ? kNH2NC : kNC2NH;
    if (InsertPostTransNode(func_graph, cnode, after_perm) != lite::RET_OK) {
      MS_LOG(ERROR) << "insert post node failed." << cnode->fullname_with_scope();
      return false;
    }
  }
  return true;
}

bool DecreaseTransposeAlgo::DecreaseTransposeForMultiOp(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto manager = Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr.";
    return false;
  }
  auto node_list = TopoSort(func_graph->get_return());
  std::set<CNodePtr> visit_transposes;
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_ASSERT(cnode != nullptr);
    if (IsSpecialType(cnode) || visit_transposes.find(cnode) != visit_transposes.end()) {
      continue;
    }
    if (CheckPrimitiveType(node, prim::kPrimIf) || CheckPrimitiveType(node, prim::kPrimWhile)) {
      auto sub_func_graph = GetValueNode<FuncGraphPtr>(cnode->input(1));
      if (sub_func_graph == nullptr) {
        return false;
      }
      (void)DecreaseTransposeForMultiOp(sub_func_graph);
      sub_func_graph = GetValueNode<FuncGraphPtr>(cnode->input(kInputIndexTwo));
      if (sub_func_graph == nullptr) {
        return false;
      }
      (void)DecreaseTransposeForMultiOp(sub_func_graph);
    }
    std::vector<int> perm{};
    if (!CheckPrimitiveType(cnode, prim::kPrimTranspose) || GetTransposePerm(cnode, &perm) != lite::RET_OK ||
        perm != kNH2NC) {
      continue;
    }
    auto status = HandleGraphMultiNode(func_graph, cnode, &visit_transposes);
    if (status != lite::RET_OK && status != lite::RET_NO_CHANGE) {
      MS_LOG(ERROR) << "global optimizer failed.";
      return false;
    }
  }
  return true;
}

bool DecreaseTransposeAlgo::RunDoFixFormat(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  auto prim_node = cnode->input(0);
  auto prim = GetValueNode<PrimitivePtr>(prim_node);
  MS_CHECK_TRUE_MSG(prim != nullptr, false, "GetValueNode Failed");
  auto &nchw_op = GetNCHWOpMap();
  if (!utils::isa<CNodePtr>(cnode->input(1))) {
    return true;
  }
  if (utils::isa<CNodePtr>(cnode->input(1))) {
    auto format = GetValue<int64_t>(prim->GetAttr(ops::kFormat));
    if (nchw_op.find(prim->name()) != nchw_op.end() && format != NCHW) {
      InsertPreTransNode(func_graph, cnode, kNH2NC);
      InsertPostTransNode(func_graph, cnode, kNC2NH);
    }
  }
  return true;
}

bool DecreaseTransposeAlgo::DoFixFormat(const FuncGraphPtr &func_graph) {
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_ASSERT(cnode != nullptr);
    if (IsSpecialType(cnode)) {
      continue;
    }
    if (CheckPrimitiveType(cnode, prim::kPrimIf) || CheckPrimitiveType(cnode, prim::kPrimWhile)) {
      auto sub_func_graph = GetValueNode<FuncGraphPtr>(cnode->input(1));
      if (sub_func_graph == nullptr) {
        lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
        return false;
      }
      SetSubGraphInput(cnode, sub_func_graph);
      if (!DoFixFormat(sub_func_graph)) {
        MS_LOG(ERROR) << "subgraph infer shape failed.";
        lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_ERROR);
        return false;
      }
      SetSubGraphOutput(sub_func_graph);

      sub_func_graph = GetValueNode<FuncGraphPtr>(cnode->input(kInputIndexTwo));
      if (sub_func_graph == nullptr) {
        lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
        return false;
      }
      SetSubGraphInput(cnode, sub_func_graph);
      if (!DoFixFormat(sub_func_graph)) {
        MS_LOG(ERROR) << "subgraph infer shape failed.";
        lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_ERROR);
        return false;
      }
      SetSubGraphOutput(sub_func_graph);
      SetSubGraphAbstract(cnode, sub_func_graph);
      continue;
    }
    if (!RunDoFixFormat(func_graph, cnode)) {
      return false;
    }
  }
  return true;
}

bool DecreaseTransposeAlgo::Run(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  node_infer_shape_.Init(fmk_type_, train_flag_);
  transpose_strategy_.Init(fmk_type_, train_flag_);
  if (!delete_redundant_transpose_.Run(func_graph)) {
    MS_LOG(ERROR) << "Run delete-redundant-transpose pass failed.";
    return false;
  }
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    auto prim = GetValueNode<PrimitivePtr>(node);
    if (prim == nullptr) {
      continue;
    }
  }

  if (!DoFixFormat(func_graph)) {
    MS_LOG(ERROR) << "DoFixFormat failed.";
    return false;
  }
  ResetSubGraphInput();

  if (!DecreaseTransposeForSingleOp(func_graph)) {
    MS_LOG(ERROR) << "run local trans insert optimizer failed.";
    return false;
  }

  auto ret = ResetSubGraphInput();
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "ResetSubGraphInput failed.";
    return false;
  }
  // if input format of several ops surrounded only by transpose op all can be NHWC,
  // we can delete these transpose ops, and at the same time, transform these middle ops.
  if (!DecreaseTransposeForMultiOp(func_graph)) {
    MS_LOG(ERROR) << "run global trans insert optimizer failed.";
    return false;
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
