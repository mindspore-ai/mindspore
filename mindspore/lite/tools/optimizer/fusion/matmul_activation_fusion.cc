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
#include "tools/optimizer/fusion/matmul_activation_fusion.h"
#include <memory>
#include "ops/fusion/activation.h"
#include "ops/fusion/mat_mul_fusion.h"
#include "include/common/utils/utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"

namespace mindspore::opt {
const BaseRef MatMulActivationFusion::DefinePattern() const {
  auto is_matmul = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMatMulFusion>);
  auto is_act = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimActivation>);
  auto act = VectorRef({is_act, is_matmul});
  return act;
}

const AnfNodePtr MatMulActivationFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                 const EquivPtr &) const {
  // Int8 MatMul Kernel dont support matmul+activation
  if (param_->commonQuantParam.quant_type == schema::QuantType_QUANT_ALL ||
      param_->commonQuantParam.quant_type == schema::QuantType_QUANT_DYNAMIC) {
    return nullptr;
  }
  if (func_graph == nullptr || node == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }
  auto act_cnode = node->cast<CNodePtr>();
  if (act_cnode == nullptr) {
    MS_LOG(ERROR) << "node is not cnode";
    return nullptr;
  }
  if (!utils::isa<CNodePtr>(act_cnode->input(1))) {
    MS_LOG(ERROR) << "matmul is not cnode.";
    return nullptr;
  }
  MS_CHECK_TRUE_RET(act_cnode->input(1) != nullptr, nullptr);
  auto matmul_cnode = act_cnode->input(1)->cast<CNodePtr>();
  auto matmul_prim = ops::GetOperator<ops::MatMulFusion>(matmul_cnode->input(0));
  if (matmul_prim == nullptr) {
    MS_LOG(ERROR) << "matmul prim is nullptr.";
    return nullptr;
  }
  auto matmul_act_type = matmul_prim->GetAttr(ops::kActivationType);
  if (matmul_act_type != nullptr && matmul_prim->get_activation_type() != ActivationType::NO_ACTIVATION) {
    MS_LOG(ERROR) << "matmul has activation.";
    return nullptr;
  }
  auto act_prim = ops::GetOperator<mindspore::ops::Activation>(act_cnode->input(0));
  if (act_prim == nullptr) {
    MS_LOG(ERROR) << "activation prim is nullptr.";
    return nullptr;
  }
  auto act_type = act_prim->GetAttr(ops::kActivationType);
  if (act_type == nullptr) {
    MS_LOG(ERROR) << "activation type attr is nullptr.";
    return nullptr;
  }
  auto type = act_prim->get_activation_type();
  if (type != mindspore::RELU && type != RELU6) {
    return nullptr;
  }
  (void)matmul_prim->AddAttr(ops::kActivationType, api::MakeValue<int64_t>(static_cast<int64_t>(type)));
  auto manage = Manage(func_graph);
  if (manage == nullptr) {
    MS_LOG(ERROR) << "manage is nullptr.";
    return nullptr;
  }
  manage->Replace(act_cnode, matmul_cnode);
  return matmul_cnode;
}
}  // namespace mindspore::opt
