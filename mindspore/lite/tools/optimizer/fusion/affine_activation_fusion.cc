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
#include "tools/optimizer/fusion/affine_activation_fusion.h"
#include <memory>
#include "tools/optimizer/common/gllo_utils.h"
#include "ops/fusion/activation.h"
#include "ops/affine.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"

namespace mindspore::opt {
const BaseRef AffineActivationFusion::DefinePattern() const {
  auto is_activation = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimActivation>);
  MS_CHECK_TRUE_RET(is_activation != nullptr, {});
  auto is_affine = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAffine>);
  MS_CHECK_TRUE_RET(is_affine != nullptr, {});
  return VectorRef({is_activation, is_affine});
}

const AnfNodePtr AffineActivationFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                 const EquivPtr &equiv) const {
  constexpr size_t kAnfPrimitiveIndex = 0;
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }
  // activation
  if (!CheckPrimitiveType(node, prim::kPrimActivation)) {
    MS_LOG(ERROR) << "the layer processed by affine fusion is not matmul.";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_PARAM_INVALID);
    return nullptr;
  }
  auto activation_node = node->cast<CNodePtr>();
  if (IsMarkedTrainOp(activation_node)) {
    return nullptr;
  }
  if (activation_node == nullptr) {
    MS_LOG(ERROR) << "the matmul_node is null.";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }
  auto activation_prim = ops::GetOperator<ops::Activation>(activation_node->input(kAnfPrimitiveIndex));
  MS_ASSERT(activation_prim != nullptr);
  AnfNodePtr pre_node = activation_node->input(1);
  if (!CheckPrimitiveType(pre_node, prim::kPrimAffine)) {
    MS_LOG(ERROR) << "previous node is not splice.";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_PARAM_INVALID);
    return nullptr;
  }
  auto affine_node = pre_node->cast<CNodePtr>();
  if (affine_node == nullptr) {
    MS_LOG(ERROR) << "the affine_node is null.";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }
  if (IsMarkedTrainOp(affine_node)) {
    return nullptr;
  }
  auto affine_prim = ops::GetOperator<ops::Affine>(affine_node->input(kAnfPrimitiveIndex));
  MS_ASSERT(affine_prim != nullptr);

  if (!activation_prim->HasAttr(ops::kActivationType)) {
    MS_LOG(ERROR) << "the kActivationType is null.";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }
  affine_prim->set_activation_type(activation_prim->get_activation_type());

  return affine_node;
}
}  // namespace mindspore::opt
