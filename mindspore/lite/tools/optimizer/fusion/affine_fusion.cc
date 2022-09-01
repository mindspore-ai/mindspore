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
#include "tools/optimizer/fusion/affine_fusion.h"
#include <memory>
#include <vector>
#include "schema/inner/model_generated.h"
#include "ops/affine.h"
#include "src/common/log_adapter.h"
#include "ops/splice.h"
#include "ops/mat_mul.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"

namespace mindspore::opt {
constexpr auto kInputWithBiasNum = 4;
constexpr auto kInputBias = 3;
const BaseRef AffineFusion::DefinePattern() const {
  auto is_matmul = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMatMulFusion>);
  MS_CHECK_TRUE_RET(is_matmul != nullptr, {});
  auto is_splice = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimSplice>);
  MS_CHECK_TRUE_RET(is_splice != nullptr, {});
  auto is_param = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_param != nullptr, {});
  auto is_seq_var = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_seq_var != nullptr, {});
  return VectorRef({is_matmul, is_splice, is_param, is_seq_var});
}

const AnfNodePtr AffineFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                       const EquivPtr &equiv) const {
  constexpr size_t kAnfPrimitiveIndex = 0;
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }
  // matmul
  if (!CheckPrimitiveType(node, prim::kPrimMatMulFusion)) {
    MS_LOG(ERROR) << "the layer processed by affine fusion is not matmul.";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_PARAM_INVALID);
    return nullptr;
  }
  auto matmul_node = node->cast<CNodePtr>();
  if (IsMarkedTrainOp(matmul_node)) {
    return nullptr;
  }
  if (matmul_node == nullptr) {
    MS_LOG(ERROR) << "the matmul_node is null.";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }
  auto matmul_prim = ops::GetOperator<ops::MatMul>(matmul_node->input(kAnfPrimitiveIndex));
  MS_CHECK_TRUE_RET(matmul_prim != nullptr, nullptr);
  auto matmul_prim_c = matmul_prim->GetPrim();
  MS_CHECK_TRUE_RET(matmul_prim_c != nullptr, nullptr);
  // splice
  AnfNodePtr pre_node = matmul_node->input(1);
  if (!CheckPrimitiveType(pre_node, prim::kPrimSplice)) {
    MS_LOG(ERROR) << "previous node is not splice.";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_PARAM_INVALID);
    return nullptr;
  }
  auto splice_node = pre_node->cast<CNodePtr>();
  if (splice_node == nullptr) {
    MS_LOG(ERROR) << "the splice_node is null.";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }
  if (IsMarkedTrainOp(splice_node)) {
    return nullptr;
  }
  auto splice_prim = ops::GetOperator<ops::Splice>(splice_node->input(kAnfPrimitiveIndex));
  MS_CHECK_TRUE_RET(splice_prim != nullptr, nullptr);
  /**
   * Affine attribute:
   * 1. context
   * 2. transpose_a
   * 3. transpose_b
   * 4. output_dim
   */
  // new primitive
  auto affine_prim = std::make_shared<ops::Affine>();
  MS_CHECK_TRUE_RET(affine_prim != nullptr, nullptr);
  auto affine_prim_c = affine_prim->GetPrim();
  MS_CHECK_TRUE_RET(affine_prim_c != nullptr, nullptr);
  // copy splice attr to affine
  MS_CHECK_TRUE_RET(splice_prim->GetAttr(ops::kSpliceContext) != nullptr, nullptr);
  affine_prim->set_context(splice_prim->get_context());
  MS_CHECK_TRUE_RET(splice_prim->GetAttr(ops::kSpliceOutputDims) != nullptr, nullptr);
  affine_prim->set_output_dim(splice_prim->get_output_dim());
  // copy matmul attribute to affine
  if (matmul_prim->GetAttr(ops::kTransposeA) != nullptr) {
    affine_prim->set_transpose_a(matmul_prim->get_transpose_a());
  }
  if (matmul_prim->GetAttr(ops::kTransposeB) != nullptr) {
    affine_prim->set_transpose_b(matmul_prim->get_transpose_b());
  }
  // construct affine node
  auto affine_value_node = NewValueNode(affine_prim_c);
  MS_CHECK_TRUE_RET(affine_value_node != nullptr, nullptr);
  std::vector<AnfNodePtr> affine_inputs = {affine_value_node, splice_node->input(1),
                                           matmul_node->input(kInputIndexTwo)};
  if (matmul_node->inputs().size() == kInputWithBiasNum) {
    affine_inputs.push_back(matmul_node->input(kInputBias));
  }
  auto affine_node = func_graph->NewCNode(affine_inputs);
  MS_CHECK_TRUE_RET(affine_node != nullptr, nullptr);
  affine_node->set_fullname_with_scope(matmul_node->fullname_with_scope());
  MS_CHECK_TRUE_RET(matmul_node->abstract() != nullptr, nullptr);
  affine_node->set_abstract(matmul_node->abstract()->Clone());

  MS_LOG(INFO) << "splice + matmul fused to affine node: " << affine_node->fullname_with_scope() << "success.";
  return affine_node;
}
}  // namespace mindspore::opt
