/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "tools/optimizer/graph/infershape_pass.h"
#include "tools/common/node_util.h"
#include "tools/common/tensor_util.h"
#include "nnacl/op_base.h"
#include "src/common/log_util.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace opt {
namespace {
int GetCNodeCertainInputFormat(const CNodePtr cnode, int index, mindspore::Format *format) {
  MS_ASSERT(cnode != nullptr && format != nullptr);
  auto origin_inputs = cnode->inputs();
  lite::RemoveIfDepend(cnode);
  lite::RemoveIfMakeTuple(cnode);
  RemoveIfMonad(cnode);
  if (index <= 0 || static_cast<size_t>(index) >= cnode->size()) {
    MS_LOG(ERROR) << "input index out of range";
    cnode->set_inputs(origin_inputs);
    return lite::RET_ERROR;
  }
  if (!utils::isa<CNode>(cnode->input(index))) {
    cnode->set_inputs(origin_inputs);
    return lite::RET_NO_CHANGE;
  }
  if (DetermineCertainVarInputFormat(cnode, index, format) != RET_OK) {
    MS_LOG(ERROR) << "determine certain var-input's  format failed.";
    return RET_ERROR;
  }
  cnode->set_inputs(origin_inputs);
  return lite::RET_OK;
}

int ModifySubGraphInputCNodeFormat(const FuncGraphPtr &sub_graph, const ParameterPtr &certain_input,
                                   mindspore::Format format) {
  MS_ASSERT(sub_graph != nullptr && certain_input != nullptr);
  auto manager = sub_graph->manager();
  MS_ASSERT(manager != nullptr);
  auto node_users = manager->node_users()[certain_input];
  for (auto &node_user : node_users) {
    if (node_user.second != 1) {
      continue;
    }
    auto post_cnode = node_user.first->cast<CNodePtr>();
    if (post_cnode == nullptr) {
      MS_LOG(ERROR) << "post node is not cnode, which is invalid.";
      return lite::RET_ERROR;
    }
    auto primitive = GetValueNode<PrimitivePtr>(post_cnode->input(0));
    MS_CHECK_TRUE_MSG(primitive != nullptr, lite::RET_NULL_PTR, "GetValueNode Failed");
    primitive->AddAttr(ops::kFormat, MakeValue<int64_t>(format));
  }
  return lite::RET_OK;
}

int JudgeControlFlowCertainOutputHasInferred(const CNodePtr &return_cnode, size_t index, bool *infer_info) {
  MS_ASSERT(return_cnode != nullptr && infer_info != nullptr);
  MS_CHECK_TRUE_MSG(index < return_cnode->size(), RET_ERROR, "input index is out of range.");
  *infer_info = true;
  auto abstract_base = GetCNodeInputAbstract(return_cnode, index);
  MS_CHECK_TRUE_MSG(abstract_base != nullptr, RET_ERROR, "anfnode has no abstract.");
  ShapeVector shape;
  auto ret = FetchShapeFromAbstract(abstract_base, &shape);
  MS_CHECK_TRUE_MSG(ret == lite::RET_OK, RET_ERROR, "fetch shape from abstract failed.");
  if (std::find(shape.begin(), shape.end(), -1) != shape.end()) {
    *infer_info = false;
    return RET_OK;
  }
  if (utils::isa<CNodePtr>(return_cnode->input(index))) {
    ret = DetermineCertainVarInputHasInferred(return_cnode, index, infer_info);
    MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "determine infer flag failed.");
  }
  return RET_OK;
}

int ModifyWhileBodyGraphInputs(const CNodePtr &cnode, const FuncGraphPtr &sub_graph, const ParameterPtr &graph_input,
                               size_t input_index) {
  MS_ASSERT(cnode != nullptr && sub_graph != nullptr && graph_input != nullptr);
  if (!CheckPrimitiveType(cnode, prim::kPrimWhile)) {
    return RET_OK;
  }
  auto body_graph = GetValueNode<FuncGraphPtr>(cnode->input(kInputIndexTwo));
  MS_ASSERT(body_graph != nullptr);
  if (body_graph.get() != sub_graph.get()) {
    MS_LOG(DEBUG) << "sub_graph is not body graph.";
    return RET_OK;
  }
  auto return_cnode = sub_graph->get_return();
  MS_CHECK_TRUE_MSG(return_cnode != nullptr, RET_ERROR, "return node is a nullptr.");
  auto origin_outputs = return_cnode->inputs();
  auto ret = lite::RemoveIfDepend(return_cnode);
  if (ret != RET_OK) {
    return_cnode->set_inputs(origin_outputs);
    MS_LOG(ERROR) << "remove depend node failed.";
    return RET_ERROR;
  }
  ret = lite::RemoveIfMakeTuple(return_cnode);
  if (ret != RET_OK) {
    return_cnode->set_inputs(origin_outputs);
    MS_LOG(ERROR) << "remove make_tuple node failed.";
    return RET_ERROR;
  }
  RemoveIfMonad(return_cnode);
  if (return_cnode->size() == 0 || input_index >= return_cnode->size() - 1) {
    return_cnode->set_inputs(origin_outputs);
    MS_LOG(ERROR) << "input index is out of range.";
    return RET_ERROR;
  }
  auto output = return_cnode->input(input_index + 1);
  return_cnode->set_inputs(origin_outputs);
  MS_CHECK_TRUE_MSG(output != nullptr, RET_ERROR, "output node is a nullptr.");
  if (output->isa<CNode>()) {
    graph_input->set_default_param(nullptr);
  }
  return RET_OK;
}

int MergeTwoBranchOfIfOp(const CNodePtr &cnode, const CNodePtr &return_cnode, size_t index, bool *true_branch) {
  MS_ASSERT(cnode != nullptr && return_cnode != nullptr && true_branch != nullptr);
  *true_branch = true;
  if (!CheckPrimitiveType(cnode, prim::kPrimIf)) {
    return RET_OK;
  }
  bool infer_info{false};
  // judge true branch.
  if (JudgeControlFlowCertainOutputHasInferred(return_cnode, index, &infer_info) != RET_OK) {
    MS_LOG(ERROR) << "determine certain output has inferred failed.";
    return RET_ERROR;
  }
  if (infer_info) {
    return RET_OK;
  }
  // judge false branch.
  if (JudgeControlFlowCertainOutputHasInferred(cnode, index + kInputSizeThree, &infer_info) != RET_OK) {
    MS_LOG(ERROR) << "determine certain output has inferred failed.";
    return RET_ERROR;
  }
  if (infer_info) {
    *true_branch = false;
  }
  return RET_OK;
}
}  // namespace

bool InferShapePass::Run(const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "func_graph is nullptr.";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return false;
  }
  node_infer_shape_ = std::make_shared<NodeInferShape>(fmk_type_, train_flag_);
  if (node_infer_shape_ == nullptr) {
    MS_LOG(ERROR) << "create NodeInferShape object failed.";
    return false;
  }
  sub_inputs_map_ = {};
  if (!JudgeAllOpsCanInfer(func_graph)) {
    MS_LOG(WARNING) << "exist op cannot support infer shape.";
    return false;
  }
  manager_ = Manage(func_graph, true);
  if (manager_ == nullptr) {
    MS_LOG(ERROR) << "generate a manager for func_graph failed.";
    return false;
  }
  if (InferProcess(func_graph) != lite::RET_OK) {
    MS_LOG(WARNING) << "infer shape failed.";
    (void)ResetSubGraphInput();
    return false;
  }
  if (ResetSubGraphInput() != lite::RET_OK) {
    MS_LOG(ERROR) << "ResetSubGraphInput failed.";
    return false;
  }
  return true;
}

bool InferShapePass::JudgeAllOpsCanInfer(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto node_list = TopoSort(func_graph->get_return());
  bool all_op_can_infer = true;
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_ASSERT(cnode != nullptr);
    if (IsSpecialType(cnode)) {
      continue;
    }
    if (lite::IsCall(cnode) || lite::IsPartialFusion(node)) {
      all_op_can_infer = false;
      return all_op_can_infer;
    }
    if (CheckPrimitiveType(node, prim::kPrimIf) || CheckPrimitiveType(node, prim::kPrimWhile)) {
      auto sub_func_graph = GetValueNode<FuncGraphPtr>(cnode->input(1));
      if (sub_func_graph == nullptr) {
        lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
        all_op_can_infer = false;
      } else {
        all_op_can_infer = all_op_can_infer && JudgeAllOpsCanInfer(sub_func_graph);
      }
      sub_func_graph = GetValueNode<FuncGraphPtr>(cnode->input(kInputIndexTwo));
      if (sub_func_graph == nullptr) {
        lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
        all_op_can_infer = false;
      } else {
        all_op_can_infer = all_op_can_infer && JudgeAllOpsCanInfer(sub_func_graph);
      }
      continue;
    }
    auto cur_op_can_infer = node_infer_shape_->JudgeOpSupportInfer(cnode);
    if (!cur_op_can_infer) {
      auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
      MS_CHECK_TRUE_MSG(prim != nullptr, false, "GetValueNode Failed");
      lite::NotSupportOp::GetInstance()->InsertOp(prim->name());
      lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NOT_SUPPORT);
      all_op_can_infer = false;
    }
  }
  return all_op_can_infer;
}

STATUS InferShapePass::InferProcess(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  manager_->AddFuncGraph(func_graph);
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNode>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_ASSERT(cnode != nullptr);
    if (IsSpecialType(cnode)) {
      continue;
    }
    if (opt::CheckPrimitiveType(node, prim::kPrimIf) || opt::CheckPrimitiveType(node, prim::kPrimWhile)) {
      auto sub_func_graph = GetValueNode<FuncGraphPtr>(cnode->input(1));
      if (sub_func_graph == nullptr) {
        lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
        return RET_ERROR;
      }
      auto ret = SetSubGraphInput(cnode, sub_func_graph);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "SetSubGraphInput failed: " << ret;
        return RET_ERROR;
      }
      if (InferProcess(sub_func_graph) != lite::RET_OK) {
        MS_LOG(WARNING) << "subgraph infer shape failed.";
        return RET_ERROR;
      }
      if (SetSubGraphOutput(sub_func_graph) != lite::RET_OK) {
        MS_LOG(ERROR) << "SetSubGraphOutput failed.";
        return RET_ERROR;
      }
      sub_func_graph = GetValueNode<FuncGraphPtr>(cnode->input(kInputIndexTwo));
      if (sub_func_graph == nullptr) {
        lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
        return RET_ERROR;
      }
      ret = SetSubGraphInput(cnode, sub_func_graph);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "SetSubGraphInput failed: " << ret;
        return RET_ERROR;
      }
      if (InferProcess(sub_func_graph) != lite::RET_OK) {
        MS_LOG(WARNING) << "subgraph infer shape failed.";
        return RET_ERROR;
      }
      if (SetSubGraphOutput(sub_func_graph) != lite::RET_OK) {
        MS_LOG(ERROR) << "SetSubGraphOutput failed.";
        return RET_ERROR;
      }
      ret = SetSubGraphAbstract(cnode, sub_func_graph);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "SetSubGraphAbstract failed: " << ret;
        return RET_ERROR;
      }
      continue;
    }
    auto status = node_infer_shape_->InferShape(cnode);
    bool infer_failed = (this->take_infer_invalid_as_failure_ && status == lite::RET_INFER_INVALID);
    infer_failed |= (status != lite::RET_OK && status != lite::RET_INFER_INVALID);
    if (infer_failed) {
      MS_LOG(WARNING) << "node infer shape failed, node is " << node->fullname_with_scope();
      return lite::RET_ERROR;
    }
    status = PostProcess(func_graph, cnode);
    if (status != lite::RET_OK) {
      MS_LOG(ERROR) << "post process current node failed, node is " << node->fullname_with_scope();
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}

STATUS InferShapePass::SetSubGraphInput(const CNodePtr &cnode, const FuncGraphPtr &sub_graph) {
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
    size_t index = 0;
    try {
      index = static_cast<size_t>(std::stoi(node_name.substr(last_underline + 1))) + kInputSizeThree;
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "Get index failed: " << e.what();
      return RET_ERROR;
    }
    auto abstract = GetCNodeInputAbstract(cnode, index);
    MS_CHECK_TRUE_MSG(abstract != nullptr, RET_ERROR, "abstract is a nullptr.");
    param_node->set_abstract(abstract->Clone());
    if (utils::isa<CNode>(cnode->input(index))) {
      ShapeVector shape_vec = {-1};
      bool has_inferred{false};
      auto ret = DetermineCertainVarInputHasInferred(cnode, index, &has_inferred);
      MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "determine infer flag failed.");
      if (!has_inferred) {
        auto abstract_shape = std::make_shared<abstract::Shape>(shape_vec);
        CHECK_NULL_RETURN(abstract_shape);
        param_node->abstract()->set_shape(abstract_shape);
      }
      mindspore::Format format = mindspore::NHWC;
      if (GetCNodeCertainInputFormat(cnode, index, &format) != lite::RET_OK) {
        MS_LOG(DEBUG) << "has no change for current control node." << cnode->fullname_with_scope();
        continue;
      }
      if (ModifySubGraphInputCNodeFormat(sub_graph, param_node, format) != lite::RET_OK) {
        MS_LOG(DEBUG) << "modify subgraph input cnode format failed." << cnode->func_graph_as_var();
      }
      continue;
    }
    if (utils::isa<Parameter>(cnode->input(index))) {
      param_node->set_default_param(cnode->input(index)->cast<ParameterPtr>()->default_param());
    }
    if (utils::isa<ValueNode>(cnode->input(index))) {
      lite::DataInfo data_info;
      auto status = lite::FetchDataFromValueNode(cnode, index, fmk_type_, train_flag_, &data_info, false);
      if (status != lite::RET_OK) {
        continue;
      }
      ShapeVector shape_vec(data_info.shape_.begin(), data_info.shape_.end());
      auto tensor_info =
        lite::CreateTensorInfo(data_info.data_.data(), data_info.data_.size(), shape_vec, (TypeId)data_info.data_type_);
      MS_CHECK_TRUE_MSG(tensor_info != nullptr, RET_ERROR, "created tensor is a nullptr.");
      param_node->set_default_param(tensor_info);
    }
    // while's body graph:if the corresponding output is a variable, the corresponding input's data will be set to NULL.
    if (ModifyWhileBodyGraphInputs(cnode, sub_graph, param_node, index - kInputSizeThree) != RET_OK) {
      MS_LOG(ERROR) << "modify while body graph's certain input failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

STATUS InferShapePass::SetSubGraphOutput(const FuncGraphPtr &sub_graph) {
  MS_ASSERT(sub_graph != nullptr);
  auto return_node = sub_graph->get_return();
  MS_ASSERT(return_node != nullptr);
  auto origin_input = return_node->inputs();
  lite::RemoveIfDepend(return_node);
  lite::RemoveIfMakeTuple(return_node);
  for (size_t i = 1; i < return_node->size(); ++i) {
    if (!opt::CheckPrimitiveType(return_node->input(i), prim::kPrimTranspose)) {
      continue;
    }
    auto node_name = return_node->input(i)->fullname_with_scope();
    if (node_name.size() < kInputSizeFive || node_name.substr(node_name.size() - kInputSizeFive) != "_post") {
      continue;
    }
    auto trans_cnode = return_node->input(i)->cast<CNodePtr>();
    MS_ASSERT(trans_cnode != nullptr);
    auto trans_input = trans_cnode->input(1);
    MS_ASSERT(trans_input != nullptr);
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

STATUS InferShapePass::SetSubGraphAbstract(const CNodePtr &cnode, const FuncGraphPtr &sub_graph) {
  MS_ASSERT(cnode != nullptr && sub_graph != nullptr);
  auto return_node = sub_graph->get_return();
  MS_ASSERT(return_node != nullptr);
  auto origin_inputs = return_node->inputs();
  lite::RemoveIfDepend(return_node);
  lite::RemoveIfMakeTuple(return_node);
  AbstractBasePtrList abstract_list;
  std::vector<bool> infer_infos;
  for (size_t i = 1; i < return_node->size(); ++i) {
    bool true_branch{false};
    auto ret = MergeTwoBranchOfIfOp(cnode, return_node, i, &true_branch);
    MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "decide to fetch which branch failed.");
    AbstractBasePtr abstract;
    bool infer_info;
    if (true_branch) {
      abstract = GetCNodeInputAbstract(return_node, i);
      if (JudgeControlFlowCertainOutputHasInferred(return_node, i, &infer_info) != lite::RET_OK) {
        MS_LOG(ERROR) << "determine certain output has inferred failed.";
        return lite::RET_ERROR;
      }
    } else {
      abstract = GetCNodeInputAbstract(cnode, i + kInputSizeThree);
      infer_info = true;
    }
    MS_CHECK_TRUE_MSG(abstract != nullptr, RET_ERROR, "get a nullptr abstract.");
    abstract_list.emplace_back(abstract->Clone());
    infer_infos.push_back(infer_info);
  }
  return_node->set_inputs(origin_inputs);
  if (utils::isa<abstract::AbstractTuplePtr>(cnode->abstract())) {
    auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(abstract_list);
    MS_CHECK_TRUE_MSG(abstract_tuple != nullptr, RET_ERROR, "created AbstractTuple is a nullptr.");
    cnode->set_abstract(abstract_tuple);
  } else {
    MS_CHECK_TRUE_MSG(abstract_list.size() == 1, RET_ERROR, "cnode output is invalid.");
    cnode->set_abstract(abstract_list.front());
  }
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_CHECK_TRUE_MSG(prim != nullptr, RET_ERROR, "cnode's input0 is not a primitive.");
  (void)prim->AddAttr(kInferFlags, MakeValue(infer_infos));
  return RET_OK;
}

int InferShapePass::ResetSubGraphInput() {
  for (auto &iter : sub_inputs_map_) {
    auto &sub_graph = iter.first;
    auto &sub_inputs = iter.second;
    MS_ASSERT(manager_ != nullptr);
    for (auto &sub_input : sub_inputs) {
      auto param_node = sub_graph->add_parameter();
      MS_CHECK_TRUE_MSG(param_node != nullptr, RET_ERROR, "Add parameter Failed");
      param_node->set_abstract(sub_input->abstract()->Clone());
      param_node->set_name(sub_input->fullname_with_scope());
      if (!manager_->Replace(sub_input, param_node)) {
        MS_LOG(ERROR) << "replace cnode failed.";
        return RET_ERROR;
      }
      auto sub_param_input = sub_input->cast<ParameterPtr>();
      MS_ASSERT(sub_param_input != nullptr);
      sub_param_input->set_default_param(nullptr);
    }
  }
  return lite::RET_OK;
}
}  // namespace opt
}  // namespace mindspore
