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

#include "tools/optimizer/graph/infershape_pass.h"
#include "tools/common/node_util.h"
#include "nnacl/op_base.h"
#include "src/common/log_util.h"

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
  auto real_cnode = cnode->input(index)->cast<CNodePtr>();
  MS_ASSERT(real_cnode != nullptr);
  if (CheckPrimitiveType(real_cnode, prim::kPrimTupleGetItem)) {
    real_cnode = real_cnode->input(1)->cast<CNodePtr>();
  }
  cnode->set_inputs(origin_inputs);
  MS_ASSERT(real_cnode != nullptr);
  auto primitive = GetValueNode<PrimitivePtr>(real_cnode->input(0));
  MS_CHECK_TRUE_MSG(primitive != nullptr, lite::RET_NULL_PTR, "GetValueNode Failed");
  if (primitive->GetAttr(ops::kFormat) == nullptr) {
    MS_LOG(ERROR) << "cnode has no format attr. " << real_cnode->fullname_with_scope();
    return lite::RET_ERROR;
  }
  auto format_attr = primitive->GetAttr(ops::kFormat);
  MS_CHECK_TRUE_MSG(format_attr != nullptr, lite::RET_NULL_PTR, "GetAttr Failed");
  *format = static_cast<mindspore::Format>(GetValue<int64_t>(format_attr));
  if (CheckPrimitiveType(real_cnode, prim::kPrimTranspose)) {
    std::vector<int> perm;
    if (GetTransposePerm(real_cnode, &perm) != lite::RET_OK) {
      MS_LOG(ERROR) << "get transpose perm failed.";
      return lite::RET_ERROR;
    }
    if (perm.size() != DIMENSION_4D) {
      return RET_OK;
    }
    if (perm == kNH2NC && *format == mindspore::NHWC) {
      *format = mindspore::NCHW;
    } else if (perm == kNC2NH && *format == mindspore::NCHW) {
      *format = mindspore::NHWC;
    }
  }
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
  if (!JudgeAllOpsCanInfer(func_graph)) {
    MS_LOG(WARNING) << "exist op cannot support infer shape.";
    return false;
  }
  if (InferProcess(func_graph) != lite::RET_OK) {
    MS_LOG(ERROR) << "infer shape failed.";
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
        return lite::RET_ERROR;
      }
      auto ret = SetSubGraphInput(cnode, sub_func_graph);
      if (ret != lite::RET_OK) {
        MS_LOG(ERROR) << "SetSubGraphInput failed: " << ret;
        return lite::RET_ERROR;
      }
      if (InferProcess(sub_func_graph) != lite::RET_OK) {
        MS_LOG(ERROR) << "subgraph infer shape failed.";
        return lite::RET_ERROR;
      }
      if (SetSubGraphOutput(sub_func_graph) != lite::RET_OK) {
        MS_LOG(ERROR) << "SetSubGraphOutput failed.";
        return lite::RET_ERROR;
      }
      sub_func_graph = GetValueNode<FuncGraphPtr>(cnode->input(kInputIndexTwo));
      if (sub_func_graph == nullptr) {
        lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
        return lite::RET_ERROR;
      }
      ret = SetSubGraphInput(cnode, sub_func_graph);
      if (ret != lite::RET_OK) {
        MS_LOG(ERROR) << "SetSubGraphInput failed: " << ret;
        return lite::RET_ERROR;
      }
      if (InferProcess(sub_func_graph) != lite::RET_OK) {
        MS_LOG(ERROR) << "subgraph infer shape failed.";
        return lite::RET_ERROR;
      }
      if (SetSubGraphOutput(sub_func_graph) != lite::RET_OK) {
        MS_LOG(ERROR) << "SetSubGraphOutput failed.";
        return lite::RET_ERROR;
      }
      ret = SetSubGraphAbstract(cnode, sub_func_graph);
      if (ret != lite::RET_OK) {
        MS_LOG(ERROR) << "SetSubGraphAbstract failed: " << ret;
        return lite::RET_ERROR;
      }
      continue;
    }
    auto status = node_infer_shape_->InferShape(cnode);
    if (status != lite::RET_OK && status != lite::RET_INFER_INVALID) {
      MS_LOG(ERROR) << "node infer shape failed, node is " << node->fullname_with_scope();
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
    auto index = 0;
    try {
      index = std::stoi(node_name.substr(last_underline + 1)) + 3;
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "Get index failed: " << e.what();
      return RET_ERROR;
    }
    param_node->set_abstract(opt::GetCNodeInputAbstract(cnode, index)->Clone());
    if (utils::isa<CNodePtr>(cnode->input(index))) {
      ShapeVector shape_vec = {-1};
      auto out_cnode = cnode->input(index)->cast<CNodePtr>();
      MS_ASSERT(trans_cnode != nullptr);
      auto out_prim = GetValueNode<PrimitivePtr>(out_cnode->input(0));
      MS_CHECK_TRUE_MSG(out_prim != nullptr, lite::RET_ERROR, "GetValueNode Failed");
      if (out_prim->GetAttr(opt::kInferDone) == nullptr || !GetValue<bool>(out_prim->GetAttr(opt::kInferDone))) {
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
        auto tensor_info = std::make_shared<tensor::Tensor>((TypeId)data_info.data_type_, shape_vec);
        CHECK_NULL_RETURN(tensor_info);
        param_node->set_default_param(tensor_info);
      } else {
        auto tensor_info = std::make_shared<tensor::Tensor>((TypeId)data_info.data_type_, shape_vec,
                                                            data_info.data_.data(), data_info.data_.size());
        CHECK_NULL_RETURN(tensor_info);
        param_node->set_default_param(tensor_info);
      }
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
  bool infer_done = true;
  for (size_t i = 1; i < return_node->size(); ++i) {
    auto abstract_base = opt::GetCNodeInputAbstract(return_node, i);
    MS_ASSERT(abstract_base != nullptr);
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
      MS_ASSERT(input_cnode != nullptr);
      if (opt::CheckPrimitiveType(input_cnode, prim::kPrimTupleGetItem)) {
        input_cnode = input_cnode->input(1)->cast<CNodePtr>();
      }
      auto input_prim = GetValueNode<PrimitivePtr>(input_cnode->input(0));
      CHECK_NULL_RETURN(input_prim);
      if (input_prim->GetAttr(opt::kInferDone) == nullptr || !GetValue<bool>(input_prim->GetAttr(opt::kInferDone))) {
        infer_done = false;
      }
    }
  }
  return_node->set_inputs(origin_inputs);
  if (utils::isa<abstract::AbstractTuplePtr>(cnode->abstract())) {
    auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(abstract_list);
    CHECK_NULL_RETURN(abstract_tuple);
    cnode->set_abstract(abstract_tuple);
  } else {
    if (abstract_list.size() != 1) {
      MS_LOG(ERROR) << "cnode output is invalid.";
    }
    cnode->set_abstract(abstract_list.front());
  }
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  CHECK_NULL_RETURN(prim);
  prim->AddAttr(opt::kInferDone, MakeValue<bool>(infer_done));
  return RET_OK;
}

int InferShapePass::ResetSubGraphInput() {
  for (auto iter = sub_inputs_map_.begin(); iter != sub_inputs_map_.end(); ++iter) {
    auto &sub_graph = iter->first;
    auto &sub_inputs = iter->second;
    auto manager = sub_graph->manager();
    MS_ASSERT(manager != nullptr);
    for (auto &sub_input : sub_inputs) {
      auto param_node = sub_graph->add_parameter();
      MS_CHECK_TRUE_MSG(param_node != nullptr, RET_ERROR, "Add parameter Failed");
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
}  // namespace opt
}  // namespace mindspore
