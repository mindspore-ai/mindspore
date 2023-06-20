/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "tools/optimizer/const_fold/fold_with_infershape.h"
#include <memory>
#include <set>
#include "mindspore/core/ops/framework_ops.h"
#include "tools/optimizer/common/format_utils.h"
#include "nnacl/op_base.h"

namespace mindspore::opt {
namespace {
constexpr auto kIsLinkWithControlFlow = "link_with_control_flow";
}  //  namespace

bool ConstFoldWithInferShape::Run(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, false);
  manager_ = Manage(func_graph);
  MS_CHECK_TRUE_RET(manager_ != nullptr, false);
  if (const_fold_processor_ == nullptr) {
    const_fold_processor_ = std::make_shared<ConstFoldProcessor>(fmk_type_, train_flag_);
  }
  MS_CHECK_TRUE_RET(const_fold_processor_ != nullptr, false);
  std::set<FuncGraphPtr> has_visited;
  if (HandleCommonFold(func_graph, &has_visited) != lite::RET_OK) {
    MS_LOG(WARNING) << "do constant fold pass failed,";
    return false;
  }
  if (HandleSpecialFold(func_graph) != lite::RET_OK) {
    MS_LOG(ERROR) << "do constant fold pass failed,";
    return false;
  }
  return true;
}

int ConstFoldWithInferShape::HandleCommonFold(const FuncGraphPtr &func_graph, std::set<FuncGraphPtr> *has_visited) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(has_visited != nullptr);
  if (has_visited->find(func_graph) != has_visited->end()) {
    return lite::RET_OK;
  }
  has_visited->insert(func_graph);
  MS_ASSERT(manager_ != nullptr);
  manager_->AddFuncGraph(func_graph);
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNode>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    for (size_t i = 0; i < cnode->size(); ++i) {
      if (IsValueNode<FuncGraph>(cnode->input(i))) {
        auto sub_graph = GetValueNode<FuncGraphPtr>(cnode->input(i));
        MS_ASSERT(sub_graph != nullptr);
        if (HandleCommonFold(sub_graph, has_visited) != lite::RET_OK) {
          MS_LOG(ERROR) << "do subgraph const-fold failed.";
          return lite::RET_ERROR;
        }
      }
    }
    if (!CheckCanCommonFold(cnode)) {
      continue;
    }
    if (const_fold_processor_->DoConstantFold(func_graph, cnode) != lite::RET_OK) {
      MS_LOG(WARNING) << "do constant fold failed.";
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}

bool ConstFoldWithInferShape::CheckCanCommonFold(const CNodePtr &cnode) const {
  MS_CHECK_TRUE_RET(cnode != nullptr, false);
  if (IsSpecialType(cnode)) {
    return false;
  }
  if (IsMarkedTrainOp(cnode) || CheckPrimitiveType(cnode, prim::kPrimCustom)) {
    return false;
  }
  auto inputs = cnode->inputs();
  return std::all_of(inputs.begin(), inputs.end(), [](const AnfNodePtr &node) {
    return (node->isa<ValueNode>() && !IsValueNode<FuncGraph>(node)) ||
           (node->isa<Parameter>() && node->cast<ParameterPtr>()->has_default());
  });
}

int ConstFoldWithInferShape::HandleSpecialFold(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  if (lite::ConverterInnerContext::GetInstance()->GetGraphInputTensorShapeMapSize() == 0) {
    return lite::RET_OK;
  }
  if (node_infershape_ == nullptr) {
    node_infershape_ = std::make_shared<NodeInferShape>(fmk_type_, train_flag_);
    MS_CHECK_TRUE_RET(node_infershape_ != nullptr, lite::RET_ERROR);
  }
  MS_ASSERT(manager_ != nullptr);
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNode>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (!CheckCanSpecialFold(cnode)) {
      continue;
    }
    if (const_fold_processor_->DoConstantFold(func_graph, cnode) != lite::RET_OK) {
      MS_LOG(ERROR) << "do constant fold failed.";
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}

bool ConstFoldWithInferShape::CheckCanSpecialFold(const CNodePtr &cnode) const {
  MS_CHECK_TRUE_RET(cnode != nullptr, false);
  for (size_t i = 0; i < cnode->size(); ++i) {
    auto input_node = cnode->input(i);
    MS_CHECK_TRUE_RET(input_node != nullptr, false);
    if (IsValueNode<FuncGraph>(input_node)) {
      return false;
    }
    if (!input_node->isa<CNode>()) {
      continue;
    }
    auto input_cnode = input_node->cast<CNodePtr>();
    auto input_prim = GetValueNode<PrimitivePtr>(input_cnode->input(0));
    MS_CHECK_TRUE_RET(input_prim != nullptr, false);
    bool is_link_with_control_flow = input_prim->GetAttr(kIsLinkWithControlFlow) == nullptr ||
                                     GetValue<bool>(input_prim->GetAttr(kIsLinkWithControlFlow));
    if (is_link_with_control_flow) {
      return false;
    }
  }
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_CHECK_TRUE_RET(prim != nullptr, false);
  prim->AddAttr(kIsLinkWithControlFlow, MakeValue(false));
  if (IsSpecialType(cnode)) {
    return false;
  }
  MS_ASSERT(node_infershape_ != nullptr);
  auto status = node_infershape_->InferShape(cnode);
  if (CheckPrimitiveType(cnode, prim::kPrimShape)) {
    return status == lite::RET_OK;
  }
  return CheckCanCommonFold(cnode);
}
}  // namespace mindspore::opt
