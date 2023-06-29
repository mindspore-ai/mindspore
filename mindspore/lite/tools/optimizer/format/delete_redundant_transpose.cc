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
#include "tools/optimizer/format/delete_redundant_transpose.h"
#include <vector>
#include "mindspore/core/ops/lite_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "tools/optimizer/common/format_utils.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"
#include "tools/common/node_util.h"
#include "tools/converter/quantizer/quant_params.h"

namespace mindspore {
namespace opt {
STATUS DeleteRedundantTranspose::DeleteControlFlowTranspose(const CNodePtr &cnode) {
  auto sub_func_graph = GetValueNode<FuncGraphPtr>(cnode->input(1));
  if (sub_func_graph == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return lite::RET_NULL_PTR;
  }
  if (DeleteNot4DTranspose(sub_func_graph) != lite::RET_OK) {
    MS_LOG(ERROR) << "delete transpose failed.";
    return lite::RET_ERROR;
  }
  sub_func_graph = GetValueNode<FuncGraphPtr>(cnode->input(kInputIndexTwo));
  if (sub_func_graph == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return lite::RET_NULL_PTR;
  }
  if (DeleteNot4DTranspose(sub_func_graph) != lite::RET_OK) {
    MS_LOG(ERROR) << "delete transpose failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

STATUS DeleteRedundantTranspose::DeleteNot4DTranspose(const FuncGraphPtr &func_graph) {
  MS_ERROR_IF_NULL_W_RET_VAL(func_graph, lite::RET_ERROR);
  MS_ERROR_IF_NULL_W_RET_VAL(manager_, lite::RET_ERROR);
  manager_->AddFuncGraph(func_graph);
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    MS_CHECK_TRUE_RET(node != nullptr, lite::RET_NULL_PTR);
    if (!utils::isa<CNode>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (CheckPrimitiveType(cnode, prim::kPrimIf) || CheckPrimitiveType(cnode, prim::kPrimWhile)) {
      if (DeleteControlFlowTranspose(cnode) != RET_OK) {
        MS_LOG(ERROR) << "DeleteControlFlowTranspose failed.";
        return lite::RET_ERROR;
      }
      continue;
    }
    if (!CheckPrimitiveType(node, prim::kPrimTranspose)) {
      continue;
    }
    auto abstract = GetCNodeInputAbstract(cnode, 1);
    ShapeVector shape;
    if (FetchShapeFromAbstract(abstract, &shape) != lite::RET_OK) {
      MS_LOG(ERROR) << "fetch shape failed.";
      return lite::RET_ERROR;
    }
    std::vector<int> perm;
    if (GetTransposePerm(cnode, &perm) != lite::RET_OK) {
      MS_LOG(ERROR) << "fetch transpose perm failed.";
      return lite::RET_ERROR;
    }
    int start_dat = 0;
    bool useless = true;
    for (auto dat : perm) {
      if (dat == start_dat) {
        start_dat += 1;
      } else {
        useless = false;
        break;
      }
    }
    if (useless) {
      if (!manager_->Replace(node, cnode->input(1))) {
        MS_LOG(ERROR) << "replace old node failed, please check.";
        return lite::RET_ERROR;
      }
      continue;
    }
    if (!lite::JudgeDynamicShape(shape) && shape.size() != perm.size()) {
      MS_LOG(DEBUG) << "transpose node need to be deleted.";
      if (UpdateNodeFormat(cnode) != lite::RET_OK) {
        MS_LOG(ERROR) << "update cnode format failed.";
        return lite::RET_ERROR;
      }
      if (!manager_->Replace(node, cnode->input(1))) {
        MS_LOG(ERROR) << "replace old node failed, please check.";
        return lite::RET_ERROR;
      }
    }
  }
  return lite::RET_OK;
}

STATUS DeleteRedundantTranspose::DoTransTransFusion(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  if (func_graph == nullptr || cnode == nullptr) {
    return lite::RET_ERROR;
  }
  if (!CheckPrimitiveType(cnode, prim::kPrimTranspose)) {
    return lite::RET_OK;
  }
  if (cnode->size() <= 1 || cnode->input(1) == nullptr) {
    MS_LOG(INFO) << "Failed to get input 1 of cnode " << cnode->fullname_with_scope() << ", input size "
                 << cnode->size();
    return lite::RET_ERROR;
  }
  auto pre_cnode = cnode->input(1)->cast<CNodePtr>();
  if (pre_cnode == nullptr) {
    MS_LOG(INFO) << "node input 1 is not a cnode, node " << cnode->fullname_with_scope();
    return lite::RET_OK;
  }
  if (!CheckPrimitiveType(pre_cnode, prim::kPrimTranspose) || IsMultiOutputTensors(func_graph, pre_cnode)) {
    return lite::RET_OK;
  }
  std::vector<int> post_perm;
  if (GetTransposePerm(cnode, &post_perm) != lite::RET_OK) {
    MS_LOG(ERROR) << "transpose perm cannot be obtained, " << cnode->fullname_with_scope();
    return lite::RET_ERROR;
  }
  std::vector<int> pre_perm;
  if (GetTransposePerm(pre_cnode, &pre_perm) != lite::RET_OK) {
    MS_LOG(ERROR) << "transpose perm cannot be obtained, " << pre_cnode->fullname_with_scope();
    return lite::RET_ERROR;
  }
  if ((pre_perm == kNH2NC && post_perm == kNC2NH) || (pre_perm == kNC2NH && post_perm == kNH2NC)) {
    auto node_users = manager_->node_users()[cnode];
    MS_LOG(INFO) << "node_users map size: " << node_users.size();
    if (!manager_->Replace(cnode, pre_cnode->input(1))) {
      MS_LOG(ERROR) << "replace old node failed, please check.";
      return lite::RET_ERROR;
    }
    if (CopyQuantParam(cnode, pre_cnode, node_users) != RET_OK) {
      MS_LOG(ERROR) << "Copy quant param failed, please check.";
      return lite::RET_ERROR;
    }
    func_graph->DropNode(cnode->input(kInputIndexTwo));
    func_graph->DropNode(pre_cnode->input(kInputIndexTwo));
  }
  return lite::RET_OK;
}

STATUS DeleteRedundantTranspose::TransTransFusion(const FuncGraphPtr &func_graph) {
  MS_ERROR_IF_NULL_W_RET_VAL(func_graph, lite::RET_ERROR);
  MS_ERROR_IF_NULL_W_RET_VAL(manager_, lite::RET_ERROR);
  manager_->AddFuncGraph(func_graph);
  auto node_lite = TopoSort(func_graph->get_return());
  for (auto &node : node_lite) {
    MS_CHECK_TRUE_RET(node != nullptr, lite::RET_NULL_PTR);
    if (!utils::isa<CNode>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (CheckPrimitiveType(cnode, prim::kPrimIf) || CheckPrimitiveType(cnode, prim::kPrimWhile)) {
      auto sub_func_graph = GetValueNode<FuncGraphPtr>(cnode->input(1));
      MS_CHECK_TRUE_MSG(sub_func_graph != nullptr, lite::RET_NULL_PTR, "find a subgraph is a nullptr.");
      if (TransTransFusion(sub_func_graph) != lite::RET_OK) {
        MS_LOG(ERROR) << "delete transpose failed.";
        return lite::RET_ERROR;
      }
      sub_func_graph = GetValueNode<FuncGraphPtr>(cnode->input(kInputIndexTwo));
      MS_CHECK_TRUE_MSG(sub_func_graph != nullptr, lite::RET_NULL_PTR, "find a subgraph is a nullptr.");
      if (TransTransFusion(sub_func_graph) != lite::RET_OK) {
        MS_LOG(ERROR) << "delete transpose failed.";
        return lite::RET_ERROR;
      }
      continue;
    }
    auto ret = DoTransTransFusion(func_graph, cnode);
    if (ret != lite::RET_OK) {
      return ret;
    }
  }
  return lite::RET_OK;
}

STATUS DeleteRedundantTranspose::UpdateNodeFormat(const CNodePtr &cnode) {
  MS_ERROR_IF_NULL_W_RET_VAL(cnode, lite::RET_ERROR);
  MS_ERROR_IF_NULL_W_RET_VAL(manager_, lite::RET_ERROR);
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_ERROR_IF_NULL_W_RET_VAL(prim, lite::RET_ERROR);
  if (prim->GetAttr(ops::kFormat) == nullptr) {
    return lite::RET_OK;
  }
  auto forward_format = GetValue<int64_t>(prim->GetAttr(ops::kFormat));
  const int max_search_depth{3};
  int loop{0};
  auto search_node = cnode->input(1);
  while (loop < max_search_depth) {
    MS_CHECK_TRUE_RET(search_node != nullptr, lite::RET_ERROR);
    auto search_cnode = search_node->cast<CNodePtr>();
    if (search_cnode == nullptr) {
      break;
    }
    auto primitive = GetCNodePrimitive(search_cnode);
    if (primitive == nullptr) {
      break;
    }
    if (primitive->GetAttr(ops::kFormat) != nullptr) {
      forward_format = GetValue<int64_t>(primitive->GetAttr(ops::kFormat));
      break;
    }
    search_node = search_cnode->input(1);
    ++loop;
  }
  auto node_users = manager_->node_users()[cnode];
  for (auto &node_user : node_users) {
    if (node_user.second != 1) {
      continue;
    }
    if (!utils::isa<CNode>(node_user.first)) {
      MS_LOG(ERROR) << "post node is not cnode, which is invalid.";
      return lite::RET_ERROR;
    }
    auto post_cnode = node_user.first->cast<CNodePtr>();
    auto post_prim = GetValueNode<PrimitivePtr>(post_cnode->input(0));
    MS_ERROR_IF_NULL_W_RET_VAL(post_prim, lite::RET_ERROR);
    post_prim->AddAttr(ops::kFormat, MakeValue<int64_t>(forward_format));
    if (prim->HasAttr(opt::kOutputsFormat)) {
      auto org_format = CastToInt(prim->GetAttr(opt::kOutputsFormat));
      std::vector<int64_t> outputs_format(org_format.size(), forward_format);
      (void)prim->AddAttr(kOutputsFormat, MakeValue(outputs_format));
    }
  }
  return lite::RET_OK;
}

bool DeleteRedundantTranspose::Run(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, false);
  manager_ = Manage(func_graph, true);
  if (manager_ == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr.";
    return false;
  }
  if (TransTransFusion(func_graph) != lite::RET_OK) {
    MS_LOG(ERROR) << "ranspose and transpose fusion failed.";
    return false;
  }
  if (DeleteNot4DTranspose(func_graph) != lite::RET_OK) {
    MS_LOG(ERROR) << "delete not 4D transpose failed.";
    return false;
  }
  return true;
}

// copy quant info from transpose to post_cnode or input_cnode
STATUS DeleteRedundantTranspose::CopyQuantParam(const CNodePtr &cnode, const CNodePtr &pre_cnode,
                                                const AnfNodeIndexSet &node_users) {
  auto input_node = pre_cnode->input(Index1);
  CHECK_NULL_RETURN(input_node);
  auto cnode_primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  CHECK_NULL_RETURN(cnode_primitive);
  auto pre_cnode_primitive = GetValueNode<PrimitivePtr>(pre_cnode->input(0));
  CHECK_NULL_RETURN(pre_cnode_primitive);
  if (lite::IsGraphInput(input_node)) {
    for (auto &node_user : node_users) {
      auto post_cnode = node_user.first->cast<CNodePtr>();
      CHECK_NULL_RETURN(post_cnode);
      auto post_cnode_primitive = GetValueNode<PrimitivePtr>(post_cnode->input(0));
      CHECK_NULL_RETURN(post_cnode_primitive);
      if (cnode_primitive->HasAttr(lite::quant::kQuantParam)) {
        auto quantization_param_value = cnode_primitive->GetAttr(lite::quant::kQuantParam);
        CHECK_NULL_RETURN(quantization_param_value);
        auto quantization_param_list = GetValue<std::vector<QuantizationParamPtr>>(quantization_param_value);
        if (!quantization_param_list.empty()) {
          MS_LOG(INFO) << "Copy quant param to " << post_cnode->fullname_with_scope();
          post_cnode_primitive->AddAttr(lite::quant::kGraphInputQuantParam, quantization_param_list.front());
        }
      }
      if (pre_cnode_primitive->HasAttr(lite::quant::kQuantParam)) {
        auto quantization_param_value = pre_cnode_primitive->GetAttr(lite::quant::kQuantParam);
        CHECK_NULL_RETURN(quantization_param_value);
        auto quantization_param_list = GetValue<std::vector<QuantizationParamPtr>>(quantization_param_value);
        if (!quantization_param_list.empty()) {
          MS_LOG(INFO) << "Copy quant param to " << post_cnode->fullname_with_scope();
          post_cnode_primitive->AddAttr(lite::quant::kGraphInputQuantParam, quantization_param_list.front());
        }
      }
    }
  } else if (input_node->isa<mindspore::CNode>()) {
    auto input_cnode = input_node->cast<mindspore::CNodePtr>();
    auto input_primitive = GetValueNode<PrimitivePtr>(input_cnode->input(0));
    CHECK_NULL_RETURN(input_primitive);
    if (cnode_primitive->HasAttr(lite::quant::kQuantParam)) {
      input_primitive->AddAttr(lite::quant::kQuantParam, cnode_primitive->GetAttr(lite::quant::kQuantParam));
    }
    if (pre_cnode_primitive->HasAttr(lite::quant::kQuantParam)) {
      input_primitive->AddAttr(lite::quant::kQuantParam, pre_cnode_primitive->GetAttr(lite::quant::kQuantParam));
    }
  } else {
    MS_LOG(ERROR) << input_node->fullname_with_scope() << " Not supported type.";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace opt
}  // namespace mindspore
