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
#include "tools/optimizer/fusion/flash_attention_fusion_for_custom.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include "nnacl/op_base.h"
#include "ops/op_utils.h"
#include "ops/batch_matmul.h"
#include "ops/fusion/flash_attention.h"
#include "ops/array_ops.h"
#include "ops/nn_ops.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "mindspore/core/ops/lite_ops.h"
#include "ops/custom.h"

namespace mindspore::opt {
namespace {
constexpr size_t kNumInputSize1 = 1;
constexpr size_t kNumInputSize2 = 2;
constexpr size_t kNumInputSize3 = 3;
}  // namespace
bool FlashAttentionFusionForCustom::InitVar() const {
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

const VectorRef FlashAttentionFusionForCustom::DefineFlashAttentionPattern1() const {
  if (!InitVar()) {
    MS_LOG(ERROR) << "initial member failed";
    return {};
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

const VectorRef FlashAttentionFusionForCustom::DefineFlashAttentionPattern2() const {
  // transpose
  auto transpose_input = std::make_shared<Var>();  // input K
  MS_CHECK_TRUE_RET(transpose_input != nullptr, {});
  auto is_transpose_param = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_transpose_param != nullptr, {});
  auto is_transpose = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimTranspose>);
  MS_CHECK_TRUE_RET(is_transpose != nullptr, {});
  auto transpose = VectorRef({is_transpose, transpose_input, is_transpose_param});
  // matmul fusion
  auto matmul_1_input = std::make_shared<Var>();  // input Q
  MS_CHECK_TRUE_RET(matmul_1_input != nullptr, {});
  auto is_matmul_1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMatMulFusion>);
  MS_CHECK_TRUE_RET(is_matmul_1 != nullptr, {});
  auto matmul_1 = VectorRef({is_matmul_1, matmul_1_input, transpose});
  // mul
  auto is_mul_param = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_mul_param != nullptr, {});
  auto is_mul = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul != nullptr, {});
  auto mul = VectorRef({is_mul, matmul_1, is_mul_param});
  // cast
  auto is_cast_1_param = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_cast_1_param != nullptr, {});
  auto is_cast_1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimCast>);
  MS_CHECK_TRUE_RET(is_cast_1 != nullptr, {});
  auto cast_1 = VectorRef({is_cast_1, mul, is_cast_1_param});
  // softmax
  auto is_softmax = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimSoftmax>);
  MS_CHECK_TRUE_RET(is_softmax != nullptr, {});
  auto softmax = VectorRef({is_softmax, cast_1});
  // cast
  auto is_cast_2_param = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_cast_2_param != nullptr, {});
  auto is_cast_2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimCast>);
  MS_CHECK_TRUE_RET(is_cast_2 != nullptr, {});
  auto cast_2 = VectorRef({is_cast_2, softmax, is_cast_2_param});
  // matmul
  auto matmul_2_input = std::make_shared<Var>();  // input Q
  MS_CHECK_TRUE_RET(matmul_2_input != nullptr, {});
  auto is_matmul_2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMatMulFusion>);
  MS_CHECK_TRUE_RET(is_matmul_2 != nullptr, {});
  auto matmul_2 = VectorRef({is_matmul_2, cast_2, matmul_2_input});
  return matmul_2;
}

const VectorRef FlashAttentionFusionForCustom::DefineFlashAttentionPattern3() const {
  // transpose
  auto transpose_input = std::make_shared<Var>();  // input K
  MS_CHECK_TRUE_RET(transpose_input != nullptr, {});
  auto is_transpose_param = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_transpose_param != nullptr, {});
  auto is_transpose = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimTranspose>);
  MS_CHECK_TRUE_RET(is_transpose != nullptr, {});
  auto transpose = VectorRef({is_transpose, transpose_input, is_transpose_param});
  // matmul fusion
  auto matmul_1_input = std::make_shared<Var>();  // input Q
  MS_CHECK_TRUE_RET(matmul_1_input != nullptr, {});
  auto is_matmul_1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMatMulFusion>);
  MS_CHECK_TRUE_RET(is_matmul_1 != nullptr, {});
  auto matmul_1 = VectorRef({is_matmul_1, matmul_1_input, transpose});
  // mul
  auto is_mul_param = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_mul_param != nullptr, {});
  auto is_mul = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul != nullptr, {});
  auto mul = VectorRef({is_mul, matmul_1, is_mul_param});
  // softmax
  auto is_softmax = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimSoftmax>);
  MS_CHECK_TRUE_RET(is_softmax != nullptr, {});
  auto softmax = VectorRef({is_softmax, mul});
  // cast
  auto is_cast_param = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_cast_param != nullptr, {});
  auto is_cast = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimCast>);
  MS_CHECK_TRUE_RET(is_cast != nullptr, {});
  auto cast = VectorRef({is_cast, softmax, is_cast_param});
  // matmul fusion
  auto matmul_2_input = std::make_shared<Var>();  // input Q
  MS_CHECK_TRUE_RET(matmul_2_input != nullptr, {});
  auto is_matmul_2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMatMulFusion>);
  MS_CHECK_TRUE_RET(is_matmul_2 != nullptr, {});
  auto matmul_2 = VectorRef({is_matmul_2, cast, matmul_2_input});
  return matmul_2;
}

std::unordered_map<std::string, VectorRef> FlashAttentionFusionForCustom::DefinePatterns() const {
  std::unordered_map<std::string, VectorRef> patterns;
  patterns["FlashAttentionPatten1"] = DefineFlashAttentionPattern1();
  patterns["FlashAttentionPatten2"] = DefineFlashAttentionPattern2();
  patterns["FlashAttentionPatten3"] = DefineFlashAttentionPattern3();
  return patterns;
}

bool FlashAttentionFusionForCustom::CheckBatchMatmulTranspose(const CNodePtr &batchmm_cnode, const bool exp_transpose_a,
                                                              const bool exp_transpose_b) const {
  auto batchmm_prim = ops::GetOperator<ops::BatchMatMul>(batchmm_cnode->input(0));
  MS_CHECK_FALSE_MSG(batchmm_prim == nullptr, false, "BatchMatMul prim is nullptr or cnode is not BatchMatMul");
  if (batchmm_prim->get_transpose_a() == exp_transpose_a && batchmm_prim->get_transpose_b() == exp_transpose_b) {
    return true;
  }
  return false;
}

bool FlashAttentionFusionForCustom::CheckInputShape(const CNodePtr &cnode, const uint32_t input_index,
                                                    const uint32_t expected_rank_num,
                                                    const uint32_t shortest_seq_len) const {
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

CNodePtr FlashAttentionFusionForCustom::CreateFlashAttentionNodePart1(const std::string &pattern_name,
                                                                      const FuncGraphPtr &func_graph,
                                                                      const AnfNodePtr &node,
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

bool FlashAttentionFusionForCustom::CheckNeedFusion(std::vector<std::string> cnode_names) const {
  if (find(plugin_custom_ops_.begin(), plugin_custom_ops_.end(), "All") != plugin_custom_ops_.end() &&
      find(plugin_custom_ops_.begin(), plugin_custom_ops_.end(), "FlashAttention") != plugin_custom_ops_.end()) {
    MS_LOG(INFO) << "can not find FA in plugin_custom_ops.";
    return false;
  }
  if (enable_pattern_names_.empty() && disable_pattern_names_.empty()) {
    return true;
  }
  if (enable_pattern_names_.find("FlashAttention") != enable_pattern_names_.end()) {
    auto enable_list = enable_pattern_names_.at("FlashAttention");
    for (auto enable_name : enable_list) {
      if (find(cnode_names.begin(), cnode_names.end(), enable_name) != cnode_names.end()) {
        return true;
      }
    }
  }
  if (disable_pattern_names_.find("FlashAttention") != disable_pattern_names_.end()) {
    auto disable_list = disable_pattern_names_.at("FlashAttention");
    for (auto disable_name : disable_list) {
      if (find(cnode_names.begin(), cnode_names.end(), disable_name) != cnode_names.end()) {
        return false;
      }
    }
    return true;
  }
  return false;
}

CNodePtr FlashAttentionFusionForCustom::CreateFlashAttentionNodePart2(const std::string &pattern_name,
                                                                      const FuncGraphPtr &func_graph,
                                                                      const AnfNodePtr &node,
                                                                      const EquivPtr &equiv) const {
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  MS_CHECK_TRUE_RET(node != nullptr, nullptr);
  MS_CHECK_TRUE_RET(equiv != nullptr, nullptr);
  std::vector<std::string> node_names = {};
  auto flash_attention_prim = std::make_shared<ops::Custom>();
  MS_CHECK_TRUE_RET(flash_attention_prim != nullptr, nullptr);
  std::vector<std::string> input_names = {"q", "k", "v"};
  std::vector<std::string> output_names = {"y"};
  flash_attention_prim->set_type("FlashAttention");
  flash_attention_prim->AddAttr("input_names", api::MakeValue(input_names));
  flash_attention_prim->AddAttr("output_names", api::MakeValue(output_names));
  flash_attention_prim->AddAttr("reg_op_name", api::MakeValue("FlashAttention"));
  auto fa_prim_c = flash_attention_prim->GetPrim();
  auto matmul_2 = node->cast<CNodePtr>();
  node_names.push_back(matmul_2->fullname_with_scope());
  MS_CHECK_TRUE_RET(matmul_2->inputs().size() >= kNumInputSize3, nullptr);
  auto cast_2 = matmul_2->input(1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(cast_2 != nullptr, nullptr);
  node_names.push_back(cast_2->fullname_with_scope());

  MS_CHECK_TRUE_RET(cast_2->inputs().size() >= kNumInputSize2, nullptr);
  auto softmax = cast_2->input(1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(softmax != nullptr, nullptr);
  node_names.push_back(softmax->fullname_with_scope());

  CNodePtr cnode = nullptr;
  if (pattern_name == "FlashAttentionPatten2") {
    MS_CHECK_TRUE_RET(softmax->inputs().size() >= kNumInputSize2, nullptr);
    cnode = softmax->input(1)->cast<CNodePtr>();
  } else if (pattern_name == "FlashAttentionPatten3") {
    cnode = softmax;
  } else {
    MS_LOG(ERROR) << "pattern name is wrong, name is: " << pattern_name;
    return nullptr;
  }
  MS_CHECK_TRUE_RET(cnode != nullptr, nullptr);
  node_names.push_back(cnode->fullname_with_scope());
  MS_CHECK_TRUE_RET(cnode->inputs().size() >= kNumInputSize2, nullptr);
  auto mul = cnode->input(1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(mul != nullptr, nullptr);
  node_names.push_back(mul->fullname_with_scope());
  MS_CHECK_TRUE_RET(mul->inputs().size() >= kNumInputSize2, nullptr);
  auto matmul_1 = mul->input(1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(matmul_1 != nullptr, nullptr);
  node_names.push_back(matmul_1->fullname_with_scope());
  MS_CHECK_TRUE_RET(matmul_1->inputs().size() >= kNumInputSize3, nullptr);
  auto transpose = matmul_1->input(kNumInputSize2)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(transpose != nullptr, nullptr);
  node_names.push_back(transpose->fullname_with_scope());
  if (!CheckNeedFusion(node_names)) {
    MS_LOG(INFO) << "not fusion this FlashAttention op.";
    return nullptr;
  }
  auto q = matmul_1->input(1);
  MS_CHECK_TRUE_RET(transpose->inputs().size() >= kNumInputSize2, nullptr);
  auto k = transpose->input(1);
  auto v = matmul_2->input(kNumInputSize2);
  MS_LOG(INFO) << "q name: " << q->fullname_with_scope() << " , k name: " << k->fullname_with_scope()
               << " , v name: " << v->fullname_with_scope();
  auto flash_attention_cnode = func_graph->NewCNode(fa_prim_c, {q, k, v});
  if (flash_attention_cnode == nullptr) {
    MS_LOG(ERROR) << "new cnode failed.";
    return nullptr;
  }
  flash_attention_cnode->set_fullname_with_scope(node->fullname_with_scope() + "_flash_attention_fusion_node");
  if (node->abstract() != nullptr) {
    flash_attention_cnode->set_abstract(node->abstract()->Clone());
  }
  return flash_attention_cnode;
}

AnfNodePtr FlashAttentionFusionForCustom::Process(const std::string &patten_name, const FuncGraphPtr &func_graph,
                                                  const AnfNodePtr &node, const EquivPtr &equiv) const {
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    MS_LOG(ERROR) << "function graph, node or equiv is nullptr.";
    return nullptr;
  }
  if (!utils::isa<CNodePtr>(node)) {
    MS_LOG(ERROR) << "this node is not cnode, node name: " << node->fullname_with_scope();
    return nullptr;
  }
  if (IsMarkedTrainOp(utils::cast<CNodePtr>(node))) {
    MS_LOG(ERROR) << "node is train op, can not fusion.";
    return nullptr;
  }
  if (patten_name == "FlashAttentionPatten2" || patten_name == "FlashAttentionPatten3") {
    auto manager = Manage(func_graph);
    auto flash_attention_node = CreateFlashAttentionNodePart2(patten_name, func_graph, node, equiv);
    if (flash_attention_node == nullptr) {
      MS_LOG(INFO) << "flash attention op not fusion.";
      return node;
    }
    manager->Replace(node, flash_attention_node);
    MS_LOG(INFO) << "flash attention fusion node name: " << node->fullname_with_scope();
    return flash_attention_node;
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
  auto cast2_cnode = add_cnode->input(kNumInputSize2)->cast<CNodePtr>();
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
  auto cnode = CreateFlashAttentionNodePart1(patten_name, func_graph, node, equiv);
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "create flash attention node failed.";
    return nullptr;
  }
  MS_LOG(INFO) << "Flash attention fusion success.";
  return cnode;
}
}  // namespace mindspore::opt
