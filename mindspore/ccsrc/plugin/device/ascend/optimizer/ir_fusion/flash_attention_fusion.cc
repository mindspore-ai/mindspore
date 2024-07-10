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
#include "plugin/device/ascend/optimizer/ir_fusion/flash_attention_fusion.h"
#include <memory>
#include "plugin/device/ascend/optimizer/common/gllo_utils.h"
#include "ops/op_utils.h"
#include "ops/array_ops.h"
#include "ops/nn_ops.h"
#include "mindspore/core/ops/lite_ops.h"
#include "ops/prompt_flash_attention.h"
#include "ops/fusion/pad_fusion.h"
#include "ops/slice.h"

namespace mindspore::opt {
namespace {
constexpr size_t kNumIndex0 = 0;
constexpr size_t kNumIndex1 = 1;
constexpr size_t kNumIndex2 = 2;
constexpr size_t kNumIndex3 = 3;
constexpr size_t kNumDimSize4 = 4;

AbstractBasePtr GetCNodeInputAbstract(const CNodePtr &cnode, size_t index) {
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "Cnode "
                  << "should not be null, but it is null.";
    return nullptr;
  }
  auto inputs = cnode->inputs();
  if (!(index > 0 && index < inputs.size())) {
    return nullptr;
  }
  auto input = inputs[index];
  if (input == nullptr) {
    MS_LOG(ERROR) << "input of Cnode " << cnode->fullname_with_scope() << "should not be null, but it is null.";
    return nullptr;
  }

  AbstractBasePtr abstract = nullptr;
  if (utils::isa<ParameterPtr>(input)) {
    auto parameter = input->cast<ParameterPtr>();
    abstract = parameter->abstract();
  } else if (utils::isa<ValueNodePtr>(input)) {
    auto value_node = input->cast<ValueNodePtr>();
    abstract = value_node->abstract();
  } else if (utils::isa<CNodePtr>(input)) {
    auto input_cnode = input->cast<CNodePtr>();
    abstract = input_cnode->abstract();
  } else {
    MS_LOG(ERROR) << "The type of input node should be ParameterPtr, ValueNodePtr or CNodePtr, "
                     "but the type of input node is not within this range.";
    return nullptr;
  }
  return abstract;
}

STATUS FetchShapeFromAbstract(const abstract::AbstractBasePtr &abstract, ShapeVector *shape) {
  if (abstract == nullptr || shape == nullptr) {
    MS_LOG(ERROR) << "The input parameters should not be null, but they are invalid.";
    return RET_ERROR;
  }
  if (!utils::isa<abstract::AbstractTensor>(abstract)) {
    MS_LOG(ERROR) << "The abstract is invalid.";
    return RET_ERROR;
  }
  auto abstract_tensor = abstract->cast<abstract::AbstractTensorPtr>();
  if (abstract_tensor->BuildShape() == nullptr || !utils::isa<abstract::ShapePtr>(abstract_tensor->BuildShape())) {
    MS_LOG(ERROR) << "The shape of abstract is invalid.";
    return RET_ERROR;
  }
  *shape = utils::cast<abstract::ShapePtr>(abstract_tensor->BuildShape())->shape();
  return RET_OK;
}

std::vector<int64_t> GetTensorShape(CNodePtr cnode, size_t input_index) {
  auto abstract = GetCNodeInputAbstract(cnode, input_index);
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "Get the input abstract of cnode " << cnode->fullname_with_scope() << "failed.";
    return {};
  }
  std::vector<int64_t> shape = {};
  if (FetchShapeFromAbstract(abstract, &shape) != RET_OK) {
    MS_LOG(ERROR) << "Fetch shape from the abstrcat of cnode " << cnode->fullname_with_scope() << "failed.";
    return {};
  }
  return shape;
}
}  // namespace

const BaseRef FlashAttentionFusion::DefinePattern() const {
  VectorRef pattern = DefineFlashAttentionPattern();
  return pattern;
}

CNodePtr FlashAttentionFusion::CreatePromptFlashAttentionCnodeForBNSD(
  const FuncGraphPtr &func_graph, const AnfNodePtr &node, const AnfNodePtr &q, const AnfNodePtr &k, const AnfNodePtr &v,
  const AnfNodePtr &atten_mask, const int64_t num_heads, const int64_t next_token, const float scale_value,
  const int64_t num_key_value_heads) const {
  MS_LOG(INFO) << "CreatePromptFlashAttentionCnodeForBNSD";
  // create op
  auto prompt_flash_attention_prim = std::make_shared<ops::PromptFlashAttention>();
  if (prompt_flash_attention_prim == nullptr) {
    MS_LOG(ERROR) << "Prompt_flash_attention_prim should not be null, but it is null.";
    return nullptr;
  }

  // add attr
  prompt_flash_attention_prim->AddAttr("num_heads", api::MakeValue(num_heads));
  prompt_flash_attention_prim->AddAttr("input_layout", api::MakeValue("BNSD"));
  prompt_flash_attention_prim->AddAttr("next_tokens", api::MakeValue(next_token));
  prompt_flash_attention_prim->AddAttr("scale_value", api::MakeValue(scale_value));
  prompt_flash_attention_prim->AddAttr("num_key_value_heads", api::MakeValue(num_key_value_heads));
  MS_LOG(INFO) << "q name: " << q->fullname_with_scope() << ", k name: " << k->fullname_with_scope()
               << ", v name: " << v->fullname_with_scope();
  MS_LOG(INFO) << "num heads" << num_heads << ", input layout: BNSD, next tokens: " << next_token
               << ", scale value:" << scale_value;

  auto fa_prim_c = prompt_flash_attention_prim->GetPrim();
  if (fa_prim_c == nullptr) {
    MS_LOG(ERROR) << "Fa_prim_c should not be null, but it is null.";
    return nullptr;
  }

  CNodePtr prompt_flash_attention_cnode = nullptr;
  if (atten_mask != nullptr) {
    prompt_flash_attention_cnode = func_graph->NewCNode(fa_prim_c, {q, k, v, atten_mask});
  } else {
    prompt_flash_attention_cnode = func_graph->NewCNode(fa_prim_c, {q, k, v});
  }
  if (prompt_flash_attention_cnode == nullptr) {
    MS_LOG(ERROR) << "New prompt_flash_attention_cnode should not be null, but it is null.";
    return nullptr;
  }
  prompt_flash_attention_cnode->set_fullname_with_scope(node->fullname_with_scope() + "_prompt_flash_attention_bnsd");
  if (node->abstract() != nullptr) {
    prompt_flash_attention_cnode->set_abstract(node->abstract()->Clone());
  }
  MS_LOG(INFO) << "Create PromptFlashAttention cnode success.";
  return prompt_flash_attention_cnode;
}

const AnfNodePtr FlashAttentionFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                               const EquivPtr &equiv) const {
  MS_LOG(INFO) << "Do FlashAttention fusion.";
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    MS_LOG(ERROR) << "Func graph, node and equiv should be not nullptr, but some of them are nullptr";
    return nullptr;
  }
  if (!utils::isa<CNodePtr>(node)) {
    MS_LOG(ERROR) << "Node should be cnode, but it is not cnode.";
    return nullptr;
  }

  auto flash_attention_node = CreateFlashAttentionNode(func_graph, node, equiv);
  if (flash_attention_node == nullptr) {
    MS_LOG(ERROR) << "FlashAttention op not fusion.";
    return nullptr;
  }
  MS_LOG(INFO) << "FlashAttention node fusion success, node name: " << flash_attention_node->fullname_with_scope();
  return flash_attention_node;
}

static int CheckInputTensorShape(const CNodePtr &matmul_1, const CNodePtr &matmul_2, const CNodePtr &atten_mask_sub,
                                 const CNodePtr &k_input_transpose) {
  auto input_tensor_q_shape = GetTensorShape(matmul_1, kNumIndex1);
  if (input_tensor_q_shape.size() != kNumDimSize4) {
    MS_LOG(ERROR) << "q shape should be 4 dims, but it is not 4 dims.";
    return RET_ERROR;
  }
  auto input_tensor_k_shape = GetTensorShape(k_input_transpose, kNumIndex1);
  if (input_tensor_k_shape.size() != kNumDimSize4) {
    MS_LOG(ERROR) << "k shape should be 4 dims, but it is not 4 dims.";
    return RET_ERROR;
  }
  auto input_tensor_v_shape = GetTensorShape(matmul_2, kNumIndex2);
  if (input_tensor_v_shape.size() != kNumDimSize4) {
    MS_LOG(ERROR) << "v shape should be 4 dims, but it is not 4 dims.";
    return RET_ERROR;
  }
  if (atten_mask_sub != nullptr) {
    auto atten_mask_shape = GetTensorShape(atten_mask_sub, kNumIndex2);
    if (atten_mask_shape.size() != kNumDimSize4) {
      MS_LOG(ERROR) << "atten_mask shape should be 4 dims, but it is not 4 dims.";
      return RET_ERROR;
    }
  }
  if (input_tensor_q_shape[kNumIndex3] <= 0 || input_tensor_q_shape[kNumIndex1] <= 0) {
    MS_LOG(ERROR) << "D should not be -1, but it is -1";
    return RET_ERROR;
  }

  return RET_OK;
}

CNodePtr FlashAttentionFusionV1::CreateFlashAttentionNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                          const EquivPtr &equiv) const {
  auto matmul_2 = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(matmul_2 != nullptr, {});
  auto softmax = matmul_2->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(softmax != nullptr, {});
  auto add = softmax->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(add != nullptr, nullptr);
  auto atten_mask_mul = add->input(kNumIndex2)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(atten_mask_mul != nullptr, nullptr);
  auto atten_mask_sub = atten_mask_mul->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(atten_mask_sub != nullptr, nullptr);
  auto realdiv = add->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(realdiv != nullptr, nullptr);
  auto matmul_1 = realdiv->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(matmul_1 != nullptr, nullptr);
  auto k_input_transpose = matmul_1->input(kNumIndex2)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(k_input_transpose != nullptr, nullptr);

  // Get FlashAttention input tensor
  auto q = matmul_1->input(kNumIndex1);
  MS_CHECK_TRUE_RET(q != nullptr, nullptr);
  auto k = k_input_transpose->input(kNumIndex1);
  MS_CHECK_TRUE_RET(k != nullptr, nullptr);
  auto v = matmul_2->input(kNumIndex2);
  MS_CHECK_TRUE_RET(v != nullptr, nullptr);
  auto atten_mask = atten_mask_sub->input(kNumIndex2);
  MS_CHECK_TRUE_RET(atten_mask != nullptr, nullptr);
  MS_LOG(INFO) << "q name: " << q->fullname_with_scope() << ", k name: " << k->fullname_with_scope()
               << ", v name: " << v->fullname_with_scope() << ", atten_mask: " << atten_mask->fullname_with_scope();

  // Check FlashAttention input tensor shape
  if (CheckInputTensorShape(matmul_1, matmul_2, atten_mask_sub, k_input_transpose) == RET_ERROR) {
    MS_LOG(ERROR) << "Check input tensor shape failed.";
    return nullptr;
  }

  auto input_tensor_q_shape = GetTensorShape(matmul_1, kNumIndex1);
  auto input_tensor_k_shape = GetTensorShape(k_input_transpose, kNumIndex1);
  int64_t seq_len = input_tensor_q_shape[kNumIndex2];
  const int64_t num_heads = input_tensor_q_shape[kNumIndex1];
  const int64_t next_token = 0;
  const float scale_value = 1 / (pow(input_tensor_q_shape[kNumIndex3], 0.5));
  const int64_t num_key_value_heads = input_tensor_k_shape[1];
  if (seq_len != 1) {
    return CreatePromptFlashAttentionCnodeForBNSD(func_graph, node, q, k, v, atten_mask, num_heads, next_token,
                                                  scale_value, num_key_value_heads);
  }
  MS_LOG(INFO) << "Seq_len is not equal to 1, do not create PromptFlashAttention cnode.";
  return nullptr;
}

const VectorRef FlashAttentionFusionV1::DefineFlashAttentionPattern() const {
  MS_LOG(INFO) << "Do FlashAttentionPattern V1.";

  // Transpose 1
  auto is_transpose_1_param = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_transpose_1_param != nullptr, {});
  auto is_transpose_1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimTranspose>);
  MS_CHECK_TRUE_RET(is_transpose_1 != nullptr, {});
  auto k_input = std::make_shared<Var>();  // input K
  MS_CHECK_TRUE_RET(k_input != nullptr, {});
  auto k_input_transpose = VectorRef({is_transpose_1, k_input, is_transpose_1_param});

  // BatchMatMul 1
  auto q_input = std::make_shared<Var>();  // input Q
  MS_CHECK_TRUE_RET(q_input != nullptr, {});
  auto is_matmul_1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimBatchMatMul>);
  MS_CHECK_TRUE_RET(is_matmul_1 != nullptr, {});
  auto matmul_1 = VectorRef({is_matmul_1, q_input, k_input_transpose});

  // RealDiv
  auto is_realdiv_param = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_realdiv_param != nullptr, {});
  auto is_realdiv = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimRealDiv>);
  MS_CHECK_TRUE_RET(is_realdiv != nullptr, {});
  auto realdiv = VectorRef({is_realdiv, matmul_1, is_realdiv_param});

  // ===== attention mask =====
  // Sub
  auto mask_sub_input = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(mask_sub_input != nullptr, {});
  auto atten_mask = std::make_shared<Var>();  // input attention mask
  MS_CHECK_TRUE_RET(atten_mask != nullptr, {});
  auto is_mask_sub = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimSub>);
  MS_CHECK_TRUE_RET(is_mask_sub != nullptr, {});
  auto atten_mask_sub = VectorRef({is_mask_sub, mask_sub_input, atten_mask});

  // Mul
  auto mask_mul_param = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(mask_mul_param != nullptr, {});
  auto is_mask_mul = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMul>);
  MS_CHECK_TRUE_RET(is_mask_mul != nullptr, {});
  auto atten_mask_mul = VectorRef({is_mask_mul, atten_mask_sub, mask_mul_param});
  // ===== end attention mask =====

  // Add
  auto is_add = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAdd>);
  MS_CHECK_TRUE_RET(is_add != nullptr, {});
  auto add = VectorRef({is_add, realdiv, atten_mask_mul});

  // Softmax
  auto dim = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(dim != nullptr, {});
  auto is_softmax = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimSoftmax>);
  MS_CHECK_TRUE_RET(is_softmax != nullptr, {});
  auto softmax = VectorRef({is_softmax, add, dim});

  // BatchMatMul 2
  auto v_input = std::make_shared<Var>();  // input V
  MS_CHECK_TRUE_RET(v_input != nullptr, {});
  auto is_matmul_2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimBatchMatMul>);
  MS_CHECK_TRUE_RET(is_matmul_2 != nullptr, {});
  auto matmul_2 = VectorRef({is_matmul_2, softmax, v_input});

  return matmul_2;
}

CNodePtr FlashAttentionFusionV2::CreateFlashAttentionNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                          const EquivPtr &equiv) const {
  auto matmul_2 = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(matmul_2 != nullptr, {});
  auto softmax = matmul_2->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(softmax != nullptr, {});
  auto add = softmax->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(add != nullptr, nullptr);
  auto mul = add->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(mul != nullptr, nullptr);
  auto matmul_1 = mul->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(matmul_1 != nullptr, nullptr);
  auto k_input_transpose = matmul_1->input(kNumIndex2)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(k_input_transpose != nullptr, nullptr);

  // Get FlashAttention input tensor
  auto q = matmul_1->input(kNumIndex1);
  MS_CHECK_TRUE_RET(q != nullptr, nullptr);
  auto k = k_input_transpose->input(kNumIndex1);
  MS_CHECK_TRUE_RET(k != nullptr, nullptr);
  auto v = matmul_2->input(kNumIndex2);
  MS_CHECK_TRUE_RET(v != nullptr, nullptr);
  auto atten_mask = nullptr;

  MS_LOG(INFO) << "q name: " << q->fullname_with_scope() << ", k name: " << k->fullname_with_scope()
               << ", v name: " << v->fullname_with_scope();

  // Check FlashAttention input tensor shape
  if (CheckInputTensorShape(matmul_1, matmul_2, nullptr, k_input_transpose) == RET_ERROR) {
    MS_LOG(ERROR) << "Check input tensor shape failed.";
    return nullptr;
  }

  auto input_tensor_q_shape = GetTensorShape(matmul_1, kNumIndex1);
  auto input_tensor_k_shape = GetTensorShape(k_input_transpose, kNumIndex1);
  int64_t seq_len = input_tensor_q_shape[kNumIndex2];
  const int64_t num_heads = input_tensor_q_shape[kNumIndex1];
  const int64_t next_token = 65535;
  const float scale_value = 1 / (pow(input_tensor_q_shape[kNumIndex3], 0.5));
  const int64_t num_key_value_heads = input_tensor_k_shape[1];
  if (seq_len != 1) {
    return CreatePromptFlashAttentionCnodeForBNSD(func_graph, node, q, k, v, atten_mask, num_heads, next_token,
                                                  scale_value, num_key_value_heads);
  }
  MS_LOG(INFO) << "Seq_len is not equal to 1, do not create PromptFlashAttention cnode.";
  return nullptr;
}

const VectorRef FlashAttentionFusionV2::DefineFlashAttentionPattern() const {
  MS_LOG(INFO) << "Do FlashAttentionPattern V2.";

  // Transpose 1
  auto is_transpose_1_param = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_transpose_1_param != nullptr, {});
  auto is_transpose_1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimTranspose>);
  MS_CHECK_TRUE_RET(is_transpose_1 != nullptr, {});
  auto k_input = std::make_shared<Var>();  // input K
  MS_CHECK_TRUE_RET(k_input != nullptr, {});
  auto k_input_transpose = VectorRef({is_transpose_1, k_input, is_transpose_1_param});

  // BatchMatMul 1
  auto q_input = std::make_shared<Var>();  // input Q
  MS_CHECK_TRUE_RET(q_input != nullptr, {});
  auto is_matmul_1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimBatchMatMul>);
  MS_CHECK_TRUE_RET(is_matmul_1 != nullptr, {});
  auto matmul_1 = VectorRef({is_matmul_1, q_input, k_input_transpose});

  // Sqrt
  auto sqrt_param = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(sqrt_param != nullptr, {});
  auto is_sqrt = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimSqrt>);
  MS_CHECK_TRUE_RET(is_sqrt != nullptr, {});
  auto sqrt = VectorRef({is_sqrt, sqrt_param});

  // RealDiv
  auto is_realdiv_param = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_realdiv_param != nullptr, {});
  auto is_realdiv = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimRealDiv>);
  MS_CHECK_TRUE_RET(is_realdiv != nullptr, {});
  auto realdiv = VectorRef({is_realdiv, is_realdiv_param, sqrt});

  // Mul
  auto is_mul = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMul>);
  MS_CHECK_TRUE_RET(is_mul != nullptr, {});
  auto mul = VectorRef({is_mul, matmul_1, realdiv});

  // FillV2
  auto is_fillv2_param_1 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_fillv2_param_1 != nullptr, {});
  auto is_fillv2_param_2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_fillv2_param_2 != nullptr, {});
  auto is_fillv2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimFillV2>);
  MS_CHECK_TRUE_RET(is_fillv2 != nullptr, {});
  auto fillv2 = VectorRef({is_fillv2, is_fillv2_param_1, is_fillv2_param_2});

  // Add
  auto is_add = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAdd>);
  MS_CHECK_TRUE_RET(is_add != nullptr, {});
  auto add = VectorRef({is_add, mul, fillv2});

  // Softmax
  auto dim = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(dim != nullptr, {});
  auto is_softmax = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimSoftmax>);
  MS_CHECK_TRUE_RET(is_softmax != nullptr, {});
  auto softmax = VectorRef({is_softmax, add, dim});

  // BatchMatMul 2
  auto v_input = std::make_shared<Var>();  // input V
  MS_CHECK_TRUE_RET(v_input != nullptr, {});
  auto is_matmul_2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimBatchMatMul>);
  MS_CHECK_TRUE_RET(is_matmul_2 != nullptr, {});
  auto matmul_2 = VectorRef({is_matmul_2, softmax, v_input});

  return matmul_2;
}

}  // namespace mindspore::opt
