/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include <memory>
#include <vector>
#include "tools/optimizer/fusion/flash_attention_fusion.h"
#include "nnacl/op_base.h"
#include "ops/fusion/activation.h"
#include "ops/op_utils.h"
#include "ops/batch_matmul.h"
#include "ops/fusion/flash_attention.h"
#include "ops/array_ops.h"
#include "ops/nn_ops.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore::opt {
const BaseRef FlashAttentionFusion::DefinePattern() const {
  if (!InitVar()) {
    MS_LOG(ERROR) << "initial member failed";
  }

  auto is_batchmm_qk = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimBatchMatMul>);
  MS_CHECK_TRUE_RET(is_batchmm_qk != nullptr, {});
  VectorRef batchmm_qk_ref({is_batchmm_qk, input_0_batchmm_qk_, input_1_batchmm_qk_});

  auto is_cast1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimCast>);
  MS_CHECK_TRUE_RET(is_cast1 != nullptr, {});
  auto cast_var = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(cast_var != nullptr, {});
  VectorRef cast_ref({is_cast1, batchmm_qk_ref, cast_var});

  // attention mask
  auto is_mul = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMul>);
  MS_CHECK_TRUE_RET(is_mul != nullptr, {});
  auto mul_var = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(mul_var != nullptr, {});
  VectorRef mul_ref({is_mul, input_0_mul_, mul_var});

  auto is_add = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAdd>);
  MS_CHECK_TRUE_RET(is_add != nullptr, {});
  VectorRef add_ref({is_add, mul_ref, cast_ref});

  auto is_softmax = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimSoftmax>);
  MS_CHECK_TRUE_RET(is_softmax != nullptr, {});
  VectorRef softmax_ref({is_softmax, add_ref});

  auto is_cast2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimCast>);
  MS_CHECK_TRUE_RET(is_cast2 != nullptr, {});
  auto cast2_var = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(cast2_var != nullptr, {});
  VectorRef cast2_ref({is_cast2, softmax_ref, cast2_var});

  auto is_batchmm_sv = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimBatchMatMul>);
  MS_CHECK_TRUE_RET(is_batchmm_sv != nullptr, {});
  VectorRef batchmm_sv_ref({is_batchmm_sv, cast2_ref, input_1_batchmm_sv_});
  return batchmm_sv_ref;
}

bool FlashAttentionFusion::InitVar() const {
  input_0_batchmm_qk_ = std::make_shared<Var>();  // tensor Q / sqrt(d)
  MS_CHECK_TRUE_RET(input_0_batchmm_qk_ != nullptr, false);
  input_1_batchmm_qk_ = std::make_shared<Var>();  // tensor K, not K.transpose
  MS_CHECK_TRUE_RET(input_1_batchmm_qk_ != nullptr, false);
  input_1_batchmm_sv_ = std::make_shared<Var>();  // tensor V
  MS_CHECK_TRUE_RET(input_1_batchmm_sv_ != nullptr, false);
  input_0_mul_ = std::make_shared<Var>();  // attention mask after sub before mul -10000
  MS_CHECK_TRUE_RET(input_0_mul_ != nullptr, false);
  return true;
}

bool FlashAttentionFusion::CheckBatchMatmulTranspose(const CNodePtr &batchmm_cnode, const bool exp_transpose_a,
                                                     const bool exp_transpose_b) const {
  auto batchmm_prim = ops::GetOperator<ops::BatchMatMul>(batchmm_cnode->input(0));
  MS_CHECK_FALSE_MSG(batchmm_prim == nullptr, false, "BatchMatMul prim is nullptr or cnode is not BatchMatMul");
  if (batchmm_prim->get_transpose_a() == exp_transpose_a && batchmm_prim->get_transpose_b() == exp_transpose_b) {
    return true;
  }
  return false;
}

bool FlashAttentionFusion::CheckInputShape(const CNodePtr &cnode, const uint32_t input_index,
                                           const uint32_t expected_rank_num, const uint32_t shortest_seq_len) const {
  auto abstract = GetCNodeInputAbstract(cnode, input_index);
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "Get abstract failed.";
    return false;
  }
  std::vector<int64_t> input_shape;
  if (FetchShapeFromAbstract(abstract, &input_shape) != lite::RET_OK) {
    return false;
  }
  const size_t seq_len_index = 2;
  auto seq_len = input_shape[seq_len_index];
  if (input_shape.size() == expected_rank_num && (seq_len == -1 || seq_len >= shortest_seq_len)) {
    return true;
  }
  return false;
}

CNodePtr FlashAttentionFusion::CreateFlashAttentionNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                        const EquivPtr &equiv) const {
  MS_ASSERT(func_graph != nullptr && node != nullptr && equiv != nullptr);
  auto fa_prim = std::make_shared<ops::FlashAttention>();
  MS_CHECK_TRUE_RET(fa_prim != nullptr, nullptr);
  auto fa_prim_c = fa_prim->GetPrim();
  MS_CHECK_TRUE_RET(fa_prim_c != nullptr, nullptr);
  auto q_node = utils::cast<AnfNodePtr>((*equiv)[input_0_batchmm_qk_]);
  MS_ASSERT(q_node != nullptr);
  auto k_node = utils::cast<AnfNodePtr>((*equiv)[input_1_batchmm_qk_]);
  MS_ASSERT(k_node != nullptr);
  auto v_node = utils::cast<AnfNodePtr>((*equiv)[input_1_batchmm_sv_]);
  MS_ASSERT(v_node != nullptr);
  auto attn_mask_node = utils::cast<AnfNodePtr>((*equiv)[input_0_mul_]);
  MS_ASSERT(attn_mask_node != nullptr);

  // if attention mask is fp32, cast it to fp16
  TypeId attn_mask_dtype;
  if (GetDataTypeFromAnfNode(attn_mask_node, &attn_mask_dtype) != RET_OK) {
    MS_LOG(ERROR) << "get input node data type failed." << attn_mask_node->fullname_with_scope();
    return nullptr;
  }
  if (!has_add_cast_ && attn_mask_dtype != kNumberTypeFloat16) {
    auto abstract = attn_mask_node->abstract();
    MS_CHECK_TRUE_RET(abstract != nullptr, nullptr);
    auto new_abstract = abstract->Clone();
    new_abstract->set_value(std::make_shared<ValueAny>());
    auto attn_mask_cnode = GenCastNode(func_graph, attn_mask_node, attn_mask_node->fullname_with_scope() + "_post_cast",
                                       static_cast<TypeId>(kNumberTypeFloat16), new_abstract);
    if (attn_mask_cnode == nullptr) {
      MS_LOG(ERROR) << "GenCastNode failed.";
      return nullptr;
    }
    attn_mask_node = attn_mask_cnode->cast<CNodePtr>();
    has_add_cast_ = True;
  }

  auto fa_cnode = func_graph->NewCNode(fa_prim_c, {q_node, k_node, v_node, attn_mask_node});
  MS_CHECK_TRUE_RET(fa_cnode != nullptr, nullptr);
  fa_cnode->set_fullname_with_scope(node->fullname_with_scope() + "_FlashAttentionFusion");
  if (node->abstract() != nullptr) {
    fa_cnode->set_abstract(node->abstract()->Clone());
  }

  // the next op is the same as qkv, which is fp16, no need to cast to fp32
  return fa_cnode;
}

const AnfNodePtr FlashAttentionFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                               const EquivPtr &equiv) const {
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    return nullptr;
  }
  if (!utils::isa<CNodePtr>(node)) {
    return nullptr;
  }
  if (IsMarkedTrainOp(utils::cast<CNodePtr>(node))) {
    return nullptr;
  }

  // check batchmm_sv transpose_a == false, transpose_b == false
  auto batchmm_sv_cnode = node->cast<CNodePtr>();
  MS_CHECK_FALSE_MSG(batchmm_sv_cnode == nullptr, nullptr, "node is not cnode.");
  if (!CheckBatchMatmulTranspose(batchmm_sv_cnode, false, false)) {
    return nullptr;
  }

  // check batchmm_qk transpose_a == false, transpose_b == true
  auto cast1_cnode = batchmm_sv_cnode->input(1)->cast<CNodePtr>();
  MS_CHECK_FALSE_MSG(cast1_cnode == nullptr, nullptr, "node is not cnode.");
  auto softmax_cnode = cast1_cnode->input(1)->cast<CNodePtr>();
  MS_CHECK_FALSE_MSG(softmax_cnode == nullptr, nullptr, "node is not cnode.");
  auto add_cnode = softmax_cnode->input(1)->cast<CNodePtr>();
  MS_CHECK_FALSE_MSG(add_cnode == nullptr, nullptr, "node is not cnode.");
  auto cast2_cnode = add_cnode->input(2)->cast<CNodePtr>();
  MS_CHECK_FALSE_MSG(cast2_cnode == nullptr, nullptr, "node is not cnode.");
  auto batchmm_qk_cnode = cast2_cnode->input(1)->cast<CNodePtr>();
  MS_CHECK_FALSE_MSG(batchmm_qk_cnode == nullptr, nullptr, "node is not cnode.");
  if (!CheckBatchMatmulTranspose(batchmm_qk_cnode, false, true)) {
    return nullptr;
  }

  // check Q, K, V, attention mask shape
  const uint32_t expected_rank_num = 4;
  const uint32_t shortest_seq_len = 32;
  const uint32_t q_batchmm_index = 1;
  const uint32_t k_batchmm_index = 2;
  const uint32_t v_batchmm_index = 2;
  const uint32_t attn_mask_mul_index = 1;
  auto mul_cnode = add_cnode->input(1)->cast<CNodePtr>();
  MS_CHECK_FALSE_MSG(mul_cnode == nullptr, nullptr, "node is not cnode.");
  if (!CheckInputShape(batchmm_qk_cnode, q_batchmm_index, expected_rank_num, shortest_seq_len) ||
      !CheckInputShape(batchmm_qk_cnode, k_batchmm_index, expected_rank_num, shortest_seq_len) ||
      !CheckInputShape(batchmm_sv_cnode, v_batchmm_index, expected_rank_num, shortest_seq_len) ||
      !CheckInputShape(mul_cnode, attn_mask_mul_index, expected_rank_num, shortest_seq_len)) {
    return nullptr;
  }

  MS_LOG(INFO) << "Flash attention matched.";
  auto cnode = CreateFlashAttentionNode(func_graph, node, equiv);
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "create flash attention node failed.";
    return nullptr;
  }
  MS_LOG(INFO) << "Flash attention fusion success.";
  return cnode;
}
}  // namespace mindspore::opt
