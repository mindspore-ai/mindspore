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

#include "tools/converter/parser/insert_transpose.h"
#include <queue>
#include <set>
#include <unordered_map>
#include <utility>
#include "ops/op_utils.h"
#include "src/common/common.h"
#include "src/common/utils.h"
#include "tools/common/tensor_util.h"

using mindspore::lite::NCHW_SHAPE;
namespace mindspore {
namespace lite {
namespace {
constexpr size_t kNCHWDimNumber = 4;
const std::vector<int> NH2NC = {0, 3, 1, 2};
const std::vector<int> NC2NH = {0, 2, 3, 1};
bool IsSpecialType(const CNodePtr &cnode) {
  if (opt::CheckPrimitiveType(cnode, prim::kPrimTupleGetItem) || opt::CheckPrimitiveType(cnode, prim::kPrimDepend) ||
      opt::CheckPrimitiveType(cnode, prim::kPrimMakeTuple) || opt::CheckPrimitiveType(cnode, opt::kPrimMakeTupleV2) ||
      opt::CheckPrimitiveType(cnode, prim::kPrimReturn)) {
    return true;
  }
  return false;
}
}  // namespace

void InsertTranspose::GetTransNodeFormatType(const CNodePtr &cnode, opt::TransTypePair *trans_info) {
  MS_ASSERT(cnode != nullptr);
  auto prim_node = cnode->input(0);
  auto prim = GetValueNode<PrimitivePtr>(prim_node);
  MS_ASSERT(prim != nullptr);
  auto &specify_nhwc_op_map = opt::GetNHWCOpMap();
  auto &specify_nchw_op_map = opt::GetNCHWOpMap();
  if (fmk_type_ == lite::converter::FmkType_TFLITE) {
    if (specify_nchw_op_map.find(prim->name()) == specify_nchw_op_map.end()) {
      return;
    }
    trans_info->pre_ = opt::kNHWC2NCHW;
    trans_info->post_ = opt::kNCHW2NHWC;
  } else if (fmk_type_ == lite::converter::FmkType_TF) {
    if (specify_nhwc_op_map.find(prim->name()) != specify_nhwc_op_map.end() && opt::GetFormat(cnode) == NCHW) {
      trans_info->pre_ = opt::kNCHW2NHWC;
      trans_info->post_ = opt::kNHWC2NCHW;
    }
    if (specify_nchw_op_map.find(prim->name()) != specify_nchw_op_map.end()) {
      trans_info->pre_ = opt::kNHWC2NCHW;
      trans_info->post_ = opt::kNCHW2NHWC;
    }
  } else {
    if (specify_nhwc_op_map.find(prim->name()) != specify_nhwc_op_map.end()) {
      if (fmk_type_ == lite::converter::FmkType_ONNX && prim->GetAttr(ops::kFormat) != nullptr &&
          GetValue<int64_t>(prim->GetAttr(ops::kFormat)) == NHWC) {
        return;
      }
      trans_info->pre_ = opt::kNCHW2NHWC;
      trans_info->post_ = opt::kNHWC2NCHW;
    }
  }
}

AnfNodePtr InsertTranspose::GenNewInputWithoutShape(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                                    const std::vector<int> &perm, bool before, size_t index) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  AnfNodePtr new_input = nullptr;
  AnfNodePtr trans_input_node = before ? cnode->input(index) : cnode;
  std::string trans_name =
    before ? cnode->fullname_with_scope() + "_pre" + std::to_string(index - 1) : cnode->fullname_with_scope() + "_post";
  new_input = opt::GenTransposeNode(func_graph, trans_input_node, perm, trans_name);
  auto new_input_prim = GetValueNode<PrimitivePtr>(new_input->cast<CNodePtr>()->input(0));
  if (perm == NC2NH) {
    new_input_prim->AddAttr(ops::kFormat, MakeValue<int64_t>(NCHW));
  } else if (perm == NH2NC) {
    new_input_prim->AddAttr(ops::kFormat, MakeValue<int64_t>(NHWC));
  }
  return new_input;
}

STATUS InsertTranspose::GenNewInput(const FuncGraphPtr &func_graph, const CNodePtr &cnode, std::vector<int> perm,
                                    bool before, size_t index) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  AnfNodePtr new_input = nullptr;

  new_input = GenNewInputWithoutShape(func_graph, cnode, perm, before, index);
  if (new_input == nullptr) {
    MS_LOG(ERROR) << "generate a transpose node failed.";
    return lite::RET_ERROR;
  }
  if (new_input == cnode->input(index) || new_input == cnode) {
    return lite::RET_OK;
  }
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    manager = Manage(func_graph, true);
  }
  MS_ASSERT(manager != nullptr);
  auto tr = manager->Transact();
  if (before) {
    tr.SetEdge(cnode, index, new_input);
    tr.Commit();
  } else {
    func_graph->manager()->Replace(cnode, new_input);
  }
  return lite::RET_OK;
}

STATUS InsertTranspose::InsertPreTransNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                           const std::vector<int> &perm) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  auto prim_node = cnode->input(0);
  auto prim = GetValueNode<PrimitivePtr>(prim_node);
  MS_ASSERT(prim != nullptr);
  auto &specify_nhwc_op_map = opt::GetNHWCOpMap();
  auto &specify_nchw_op_map = opt::GetNCHWOpMap();
  if (specify_nhwc_op_map.find(prim->name()) == specify_nhwc_op_map.end() &&
      specify_nchw_op_map.find(prim->name()) == specify_nchw_op_map.end()) {
    MS_LOG(ERROR) << "op don't meet nhwc condition.";
    return lite::RET_ERROR;
  }
  std::vector<size_t> insert_index = specify_nchw_op_map.find(prim->name()) == specify_nchw_op_map.end()
                                       ? specify_nhwc_op_map.at(prim->name())
                                       : specify_nchw_op_map.at(prim->name());
  if (insert_index.empty()) {
    if (opt::CheckPrimitiveType(cnode, prim::kPrimResizeGrad) && prim->GetAttr(ops::kMethod) != nullptr &&
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

STATUS InsertTranspose::InsertPostTransNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
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
      if (!opt::CheckPrimitiveType(post_node, prim::kPrimTupleGetItem)) {
        if (!train_flag_) {
          MS_LOG(ERROR) << "post node is invalid.";
          return lite::RET_ERROR;
        } else {
          tuple_get_item = opt::GenTupleGetItemNode(func_graph, cnode, 0);
          post_node = tuple_get_item;
          func_graph->manager()->Replace(cnode, tuple_get_item);
        }
      }
      if (func_graph->manager()->node_users()[post_node].empty()) {
        continue;
      }
      auto post_cnode = post_node->cast<CNodePtr>();
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

STATUS InsertTranspose::HandleGraphInput(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  if (fmk_type_ == lite::converter::FmkType_TF || fmk_type_ == lite::converter::FmkType_TFLITE) {
    return lite::RET_NO_CHANGE;
  }
  for (size_t i = 1; i < cnode->size(); ++i) {
    auto node = cnode->input(i);
    if (!utils::isa<ParameterPtr>(node)) {
      continue;
    }
    auto param_node = node->cast<ParameterPtr>();
    if (param_node->has_default()) {
      continue;
    }
    auto abstract_base = param_node->abstract();
    if (abstract_base == nullptr) {
      MS_LOG(ERROR) << "Abstract of parameter is nullptr, " << param_node->name();
      return lite::RET_ERROR;
    }
    if (!utils::isa<abstract::AbstractTensorPtr>(abstract_base)) {
      MS_LOG(ERROR) << "Abstract of parameter should be anstract tensor, " << param_node->name();
      return lite::RET_ERROR;
    }
    auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(abstract_base);
    if (!utils::isa<abstract::ShapePtr>(abstract_tensor->BuildShape())) {
      MS_LOG(ERROR) << "Shape of Abstract of parameter should be ShapePtr, " << param_node->name();
      return lite::RET_ERROR;
    }
    auto shape_vector = utils::cast<abstract::ShapePtr>(abstract_tensor->BuildShape())->shape();
    if (shape_vector.size() != 4) {
      continue;
    }
    if (func_graph->get_inputs().size() == 1 && fmk_type_ == lite::converter::FmkType_ONNX && shape_vector[3] == 3 &&
        shape_vector[1] == -1) {
      continue;
    }
    std::vector<int64_t> new_dims = {shape_vector[NCHW_SHAPE::NCHW_N], shape_vector[NCHW_SHAPE::NCHW_H],
                                     shape_vector[NCHW_SHAPE::NCHW_W], shape_vector[NCHW_SHAPE::NCHW_C]};
    abstract_tensor->set_shape(std::make_shared<abstract::Shape>(new_dims));
    auto trans_cnode = opt::GenTransposeNode(func_graph, param_node, NH2NC, param_node->fullname_with_scope() + "_pre");
    auto new_input_prim = GetValueNode<PrimitivePtr>(trans_cnode->cast<CNodePtr>()->input(0));
    new_input_prim->AddAttr(ops::kFormat, MakeValue<int64_t>(NHWC));
    if (trans_cnode == nullptr) {
      MS_LOG(ERROR) << "generate a transpose node failed.";
      return lite::RET_ERROR;
    }
    func_graph->manager()->Replace(param_node, trans_cnode);
  }
  return lite::RET_OK;
}

STATUS InsertTranspose::HandleGraphNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  opt::TransTypePair trans_info;
  GetTransNodeFormatType(cnode, &trans_info);
  if (trans_info.pre_ == opt::kNONE || trans_info.post_ == opt::kNONE) {
    return lite::RET_NO_CHANGE;
  }
  auto before_perm = trans_info.pre_ == opt::kNHWC2NCHW ? NH2NC : NC2NH;
  auto after_perm = trans_info.post_ == opt::kNCHW2NHWC ? NC2NH : NH2NC;
  if (InsertPreTransNode(func_graph, cnode, before_perm) != lite::RET_OK) {
    MS_LOG(ERROR) << "insert pre node failed." << cnode->fullname_with_scope();
    return lite::RET_ERROR;
  }
  if (opt::CheckPrimitiveType(cnode, prim::kPrimAdam) || opt::CheckPrimitiveType(cnode, prim::kPrimSGD)) {
    return RET_OK;
  }
  if (InsertPostTransNode(func_graph, cnode, after_perm) != lite::RET_OK) {
    MS_LOG(ERROR) << "insert post node failed." << cnode->fullname_with_scope();
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

void InsertTranspose::SetSubGraphInput(const CNodePtr &cnode, const FuncGraphPtr &sub_graph) {
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
    auto index = std::stoi(node_name.substr(last_underline + 1)) + 3;
    param_node->set_abstract(opt::GetCNodeInputAbstract(cnode, index)->Clone());
    if (utils::isa<CNodePtr>(cnode->input(index))) {
      ShapeVector shape_vec = {-1};
      auto out_cnode = cnode->input(index)->cast<CNodePtr>();
      MS_ASSERT(trans_cnode != nullptr);
      auto out_prim = GetValueNode<PrimitivePtr>(out_cnode->input(0));
      if (out_prim->GetAttr(opt::kInferDone) == nullptr || !GetValue<bool>(out_prim->GetAttr(opt::kInferDone))) {
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
}

void InsertTranspose::ResetSubGraphInput() {
  for (auto iter = sub_inputs_map_.begin(); iter != sub_inputs_map_.end(); ++iter) {
    auto &sub_graph = iter->first;
    auto &sub_inputs = iter->second;
    auto manager = sub_graph->manager();
    MS_ASSERT(manager != nullptr);
    for (auto &sub_input : sub_inputs) {
      auto param_node = sub_graph->add_parameter();
      MS_ASSERT(param_node != nullptr);
      param_node->set_abstract(sub_input->abstract()->Clone());
      param_node->set_name(sub_input->fullname_with_scope());
      manager->Replace(sub_input, param_node);
      auto sub_param_input = sub_input->cast<ParameterPtr>();
      MS_ASSERT(sub_param_input != nullptr);
      sub_param_input->set_default_param(nullptr);
    }
  }
}

void InsertTranspose::SetSubGraphOutput(const CNodePtr &cnode, const FuncGraphPtr &sub_graph) {
  MS_ASSERT(cnode != nullptr && sub_graph != nullptr);
  auto return_node = sub_graph->get_return();
  auto origin_input = return_node->inputs();
  lite::RemoveIfDepend(return_node);
  lite::RemoveIfMakeTuple(return_node);
  for (size_t i = 1; i < return_node->size(); ++i) {
    if (!opt::CheckPrimitiveType(return_node->input(i), prim::kPrimTranspose)) {
      continue;
    }
    auto node_name = return_node->input(i)->fullname_with_scope();
    if (node_name.substr(node_name.size() - 5) != "_post") {
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
}

void InsertTranspose::SetSubGraphAbstract(const CNodePtr &cnode, const FuncGraphPtr &sub_graph) {
  MS_ASSERT(cnode != nullptr && sub_graph != nullptr);
  auto return_node = sub_graph->get_return();
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
      if (opt::CheckPrimitiveType(input_cnode, prim::kPrimTupleGetItem)) {
        input_cnode = input_cnode->input(1)->cast<CNodePtr>();
      }
      auto input_prim = GetValueNode<PrimitivePtr>(input_cnode->input(0));
      if (input_prim->GetAttr(opt::kInferDone) == nullptr || !GetValue<bool>(input_prim->GetAttr(opt::kInferDone))) {
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
  prim->AddAttr(opt::kInferDone, MakeValue<bool>(infer_done));
}

bool InsertTranspose::BasicProcess(const FuncGraphPtr &func_graph, bool main_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto graph_name = GetValue<std::string>(func_graph->get_attr("graph_name"));
  auto manager = Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr.";
    return false;
  }
  auto node_list = TopoSort(func_graph->get_return());
  int status;
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (IsSpecialType(cnode)) {
      continue;
    }
    if (main_graph) {
      status = HandleGraphInput(func_graph, cnode);
      if (status != lite::RET_OK && status != lite::RET_NO_CHANGE) {
        return false;
      }
    }
    if (opt::CheckPrimitiveType(node, prim::kPrimIf) || opt::CheckPrimitiveType(node, prim::kPrimWhile)) {
      auto sub_func_graph = GetValueNode<FuncGraphPtr>(cnode->input(1));
      if (sub_func_graph == nullptr) {
        lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
        return false;
      }
      SetSubGraphInput(cnode, sub_func_graph);
      (void)BasicProcess(sub_func_graph, false);
      SetSubGraphOutput(cnode, sub_func_graph);
      sub_func_graph = GetValueNode<FuncGraphPtr>(cnode->input(2));
      if (sub_func_graph == nullptr) {
        lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
        return false;
      }
      SetSubGraphInput(cnode, sub_func_graph);
      (void)BasicProcess(sub_func_graph, false);
      SetSubGraphOutput(cnode, sub_func_graph);
      SetSubGraphAbstract(cnode, sub_func_graph);
      continue;
    }
    status = HandleGraphNode(func_graph, cnode);
    if (status != lite::RET_OK && status != lite::RET_NO_CHANGE) {
      return false;
    }
  }
  return true;
}

bool InsertTranspose::ResetFuncGraph(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto manager = Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr.";
    return false;
  }
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (prim->GetAttr(opt::kInferDone) != nullptr) {
      prim->EraseAttr(opt::kInferDone);
    }
    if (opt::CheckPrimitiveType(node, prim::kPrimIf) || opt::CheckPrimitiveType(node, prim::kPrimWhile)) {
      auto sub_func_graph = GetValueNode<FuncGraphPtr>(cnode->input(1));
      if (sub_func_graph == nullptr) {
        return false;
      }
      (void)ResetFuncGraph(sub_func_graph);
      sub_func_graph = GetValueNode<FuncGraphPtr>(cnode->input(2));
      if (sub_func_graph == nullptr) {
        return false;
      }
      (void)ResetFuncGraph(sub_func_graph);
    }
  }
  return true;
}

bool InsertTranspose::Run(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    auto prim = GetValueNode<PrimitivePtr>(node);
    if (prim == nullptr) {
      continue;
    }
  }
  // insert transpose for some ops whose format must be NHWC, which is depend on framework.
  // In this process, tranpose can be fused, which the original graph may not be able to restored.
  if (!BasicProcess(func_graph, true)) {
    MS_LOG(ERROR) << "run framework transpose unify failed.";
    return false;
  }
  ResetSubGraphInput();
  return true;
}
}  // namespace lite
}  // namespace mindspore
