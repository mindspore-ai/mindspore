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
#include "tools/optimizer/fusion/add_concat_activation_fusion.h"
#include <memory>
#include "ops/concat.h"
#include "ops/fusion/activation.h"
#include "ops/fusion/add_fusion.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"

namespace mindspore::opt {
const BaseRef AddConcatActivationFusion::DefinePattern() const {
  auto is_act = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimActivation>);
  MS_CHECK_TRUE_RET(is_act != nullptr, {});
  auto is_concat = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimConcat>);
  MS_CHECK_TRUE_RET(is_concat != nullptr, {});
  auto is_seq_var = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_seq_var != nullptr, {});
  VectorRef pattern_ref = VectorRef({is_act, is_concat, is_seq_var});
  return pattern_ref;
}

const AnfNodePtr AddConcatActivationFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                    const EquivPtr &) const {
  if (func_graph == nullptr || node == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }
  if (!CheckPrimitiveType(node, prim::kPrimActivation)) {
    MS_LOG(INFO) << "node is not activation node";
    return nullptr;
  }
  auto act_cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(act_cnode != nullptr, nullptr);
  auto concat_node = act_cnode->input(1);
  auto concat_cnode = concat_node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(concat_node != nullptr, nullptr);
  if (concat_cnode->size() != kInputIndexThree || !utils::isa<CNode>(concat_cnode->input(kInputIndexTwo))) {
    MS_LOG(INFO) << "concat node must link two add node in front";
    return nullptr;
  }
  auto right_add_node = concat_cnode->input(1);
  MS_CHECK_TRUE_RET(right_add_node != nullptr, nullptr);
  if (!CheckPrimitiveType(right_add_node, prim::kPrimAddFusion)) {
    MS_LOG(INFO) << "right node is not add node";
    return nullptr;
  }
  auto right_add_cnode = right_add_node->cast<CNodePtr>();
  auto right_add_prim = ops::GetOperator<ops::AddFusion>(right_add_cnode->input(0));
  MS_CHECK_TRUE_RET(right_add_prim != nullptr, nullptr);
  if (right_add_prim->GetAttr(ops::kActivationType) == nullptr) {
    right_add_prim->AddAttr(ops::kActivationType,
                            api::MakeValue<int64_t>(static_cast<int64_t>(ActivationType::NO_ACTIVATION)));
  }
  if (right_add_prim->get_activation_type() != ActivationType::NO_ACTIVATION) {
    MS_LOG(INFO) << "right add node has activation";
    return nullptr;
  }

  auto left_add_node = concat_cnode->input(kInputIndexTwo);
  MS_CHECK_TRUE_RET(left_add_node != nullptr, nullptr);
  if (!CheckPrimitiveType(left_add_node, prim::kPrimAddFusion)) {
    return nullptr;
  }
  auto left_add_cnode = left_add_node->cast<CNodePtr>();
  auto left_add_prim = ops::GetOperator<ops::AddFusion>(left_add_cnode->input(0));
  MS_CHECK_TRUE_RET(left_add_prim != nullptr, nullptr);
  if (left_add_prim->GetAttr(ops::kActivationType) == nullptr) {
    left_add_prim->AddAttr(ops::kActivationType,
                           api::MakeValue<int64_t>(static_cast<int64_t>(ActivationType::NO_ACTIVATION)));
  }
  if (left_add_prim->get_activation_type() != ActivationType::NO_ACTIVATION) {
    MS_LOG(INFO) << "left add node has activation";
    return nullptr;
  }
  auto act_prim = ops::GetOperator<ops::Activation>(act_cnode->input(0));
  MS_CHECK_TRUE_RET(act_prim != nullptr, nullptr);
  if (act_prim->GetAttr(ops::kActivationType) != nullptr) {
    right_add_prim->set_activation_type(act_prim->get_activation_type());
    left_add_prim->set_activation_type(act_prim->get_activation_type());
  }

  // delete activation node
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_RET(manager != nullptr, nullptr);
  (void)manager->Replace(act_cnode, act_cnode->input(1));

  return nullptr;
}
}  // namespace mindspore::opt
