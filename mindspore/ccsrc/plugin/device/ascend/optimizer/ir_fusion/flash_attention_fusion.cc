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
#include "plugin/device/ascend/optimizer/ir_fusion/flash_attention_fusion.h"
#include <memory>
#include "plugin/device/ascend/optimizer/common/gllo_utils.h"
#include "ops/op_utils.h"
#include "ops/array_ops.h"
#include "ops/nn_ops.h"
#include "mindspore/core/ops/lite_ops.h"
#include "ops/incre_flash_attention.h"
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
constexpr size_t kNumShapeSize4 = 4;
constexpr int64_t kNumMinSeqLenSize = 1024;
constexpr int64_t kNumMaxSeqLenSize = 4096;
constexpr int64_t kNumMaxNextTokenSize = 65535;
constexpr int kNumMultiple32 = 32;
constexpr int kNumMultiple16 = 16;
constexpr int64_t kNumDValue = 40;
constexpr int64_t kNumPadSize = 8;

AbstractBasePtr GetCNodeInputAbstract(const CNodePtr &cnode, size_t index) {
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "CNodePtr is nullptr";
    return nullptr;
  }
  auto inputs = cnode->inputs();
  if (!(index > 0 && index < inputs.size())) {
    return nullptr;
  }
  auto input = inputs[index];
  if (input == nullptr) {
    MS_LOG(ERROR) << "CNode input is nullptr";
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
    MS_LOG(ERROR) << "unsupported input node type";
    return nullptr;
  }
  return abstract;
}

STATUS FetchShapeFromAbstract(const abstract::AbstractBasePtr &abstract, ShapeVector *shape) {
  if (abstract == nullptr || shape == nullptr) {
    MS_LOG(ERROR) << "input parameter is nullptr, which is invalid.";
    return RET_ERROR;
  }
  if (!utils::isa<abstract::AbstractTensor>(abstract)) {
    MS_LOG(ERROR) << "abstract of cnode is invalid.";
    return RET_ERROR;
  }
  auto abstract_tensor = abstract->cast<abstract::AbstractTensorPtr>();
  if (abstract_tensor->BuildShape() == nullptr || !utils::isa<abstract::ShapePtr>(abstract_tensor->BuildShape())) {
    MS_LOG(ERROR) << "shape of cnode's output is invalid.";
    return RET_ERROR;
  }
  *shape = utils::cast<abstract::ShapePtr>(abstract_tensor->BuildShape())->shape();
  return RET_OK;
}

std::vector<int64_t> GetTensorShape(CNodePtr cnode, size_t input_index) {
  auto abstract = GetCNodeInputAbstract(cnode, input_index);
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "GetCNodeInputAbstract in promapt flash attention fusion.";
    return {};
  }
  std::vector<int64_t> shape = {};
  if (FetchShapeFromAbstract(abstract, &shape) != RET_OK) {
    MS_LOG(ERROR) << "FetchShapeFromAbstract failed.";
    return {};
  }
  return shape;
}
}  // namespace

const VectorRef FlashAttentionFusion::DefineFlashAttentionPattern() const {
  MS_LOG(INFO) << "Do FlashAttentionPattern.";
  // Transpose 1
  auto is_transpose_1_param = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_transpose_1_param != nullptr, {});
  auto is_transpose_1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimTranspose>);
  MS_CHECK_TRUE_RET(is_transpose_1 != nullptr, {});
  auto matmul_1_k_input = std::make_shared<Var>();  // input K
  MS_CHECK_TRUE_RET(matmul_1_k_input != nullptr, {});
  auto matmul_1_k_input_transpose = VectorRef({is_transpose_1, matmul_1_k_input, is_transpose_1_param});

  // BatchMatMul 1
  auto matmul_1_q_input = std::make_shared<Var>();  // input Q
  MS_CHECK_TRUE_RET(matmul_1_q_input != nullptr, {});
  auto is_matmul_1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimBatchMatMul>);
  MS_CHECK_TRUE_RET(is_matmul_1 != nullptr, {});
  auto matmul_1 = VectorRef({is_matmul_1, matmul_1_q_input, matmul_1_k_input_transpose});

  // RealDiv
  auto is_realdiv_param = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_realdiv_param != nullptr, {});
  auto is_realdiv = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimRealDiv>);
  MS_CHECK_TRUE_RET(is_realdiv != nullptr, {});
  auto realdiv = VectorRef({is_realdiv, matmul_1, is_realdiv_param});

  // ===== attention mask =====
  // Sub
  auto sub_mask_input_1 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(sub_mask_input_1 != nullptr, {});
  auto atten_mask = std::make_shared<Var>();  // input attention mask
  MS_CHECK_TRUE_RET(atten_mask != nullptr, {});
  auto is_mask_sub = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimSub>);
  MS_CHECK_TRUE_RET(is_mask_sub != nullptr, {});
  auto atten_mask_sub = VectorRef({is_mask_sub, sub_mask_input_1, atten_mask});

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
  auto is_softmax = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimSoftmax>);
  MS_CHECK_TRUE_RET(is_softmax != nullptr, {});
  auto softmax = VectorRef({is_softmax, add});

  // BatchMatMul 2
  auto v_input = std::make_shared<Var>();  // input V
  MS_CHECK_TRUE_RET(v_input != nullptr, {});
  auto is_matmul_2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimBatchMatMul>);
  MS_CHECK_TRUE_RET(is_matmul_2 != nullptr, {});
  auto matmul_2 = VectorRef({is_matmul_2, softmax, v_input});

  return matmul_2;
}

const BaseRef FlashAttentionFusion::DefinePattern() const {
  VectorRef pattern = DefineFlashAttentionPattern();
  return pattern;
}

CNodePtr FlashAttentionFusion::CreatePromptFlashAttentionCnodeForBNSD(const FuncGraphPtr &func_graph,
                                                                      const AnfNodePtr &node, const AnfNodePtr &q,
                                                                      const AnfNodePtr &k, const AnfNodePtr &v,
                                                                      const AnfNodePtr &atten_mask, int64_t num_heads,
                                                                      int64_t next_token, float scale_value,
                                                                      int64_t num_key_value_heads) const {
  // create op
  auto prompt_flash_attention_prim = std::make_shared<ops::PromptFlashAttention>();
  if (prompt_flash_attention_prim == nullptr) {
    MS_LOG(ERROR) << "New prompt flash attention prim failed.";
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
    MS_LOG(ERROR) << "fa_prim_c is nullptr.";
    return nullptr;
  }

  CNodePtr prompt_flash_attention_cnode = nullptr;
  if (atten_mask != nullptr) {
    prompt_flash_attention_cnode = func_graph->NewCNode(fa_prim_c, {q, k, v, atten_mask, atten_mask});
  } else {
    prompt_flash_attention_cnode = func_graph->NewCNode(fa_prim_c, {q, k, v});
  }
  if (prompt_flash_attention_cnode == nullptr) {
    MS_LOG(ERROR) << "New cnode failed.";
    return nullptr;
  }
  prompt_flash_attention_cnode->set_fullname_with_scope(node->fullname_with_scope() + "_prompt_flash_attention_bnsd");
  if (node->abstract() != nullptr) {
    prompt_flash_attention_cnode->set_abstract(node->abstract()->Clone());
  }
  return prompt_flash_attention_cnode;
}

CNodePtr FlashAttentionFusion::CreateFlashAttentionNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
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
  auto matmul_1_k_input_transpose = matmul_1->input(kNumIndex2)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(matmul_1_k_input_transpose != nullptr, nullptr);

  // Get FlashAttention input tensor
  auto q = matmul_1->input(kNumIndex1);
  MS_CHECK_TRUE_RET(q != nullptr, nullptr);
  auto k = matmul_1_k_input_transpose->input(kNumIndex1);
  MS_CHECK_TRUE_RET(k != nullptr, nullptr);
  auto v = matmul_2->input(kNumIndex2);
  MS_CHECK_TRUE_RET(v != nullptr, nullptr);
  auto atten_mask = atten_mask_sub->input(kNumIndex2);
  MS_CHECK_TRUE_RET(atten_mask != nullptr, nullptr);
  MS_LOG(INFO) << "q name: " << q->fullname_with_scope() << ", k name: " << k->fullname_with_scope()
               << ", v name: " << v->fullname_with_scope() << ", atten_mask: " << atten_mask->fullname_with_scope();

  // Get FlashAttention input tensor shape
  auto input_tensor_q_shape = GetTensorShape(matmul_1, kNumIndex1);
  if (input_tensor_q_shape.size() != kNumDimSize4) {
    MS_LOG(ERROR) << "q shape is not 4 dims";
    return nullptr;
  }
  auto input_tensor_k_shape = GetTensorShape(matmul_1_k_input_transpose, kNumIndex1);
  if (input_tensor_k_shape.size() != kNumDimSize4) {
    MS_LOG(ERROR) << "k shape is not 4 dims";
    return nullptr;
  }
  auto input_tensor_v_shape = GetTensorShape(matmul_2, kNumIndex2);
  if (input_tensor_v_shape.size() != kNumDimSize4) {
    MS_LOG(ERROR) << "v shape is not 4 dims";
    return nullptr;
  }
  auto atten_mask_shape = GetTensorShape(atten_mask_sub, kNumIndex2);
  if (atten_mask_shape.size() != kNumDimSize4) {
    MS_LOG(ERROR) << "atten_mask_shape shape is not 4 dims";
    return nullptr;
  }

  MS_LOG(INFO) << "q shape: " << input_tensor_q_shape << ", k shape: " << input_tensor_k_shape << ", v shape"
               << input_tensor_v_shape << ", atten_mask shape: " << atten_mask_shape;

  // Check input tensor shape
  if (input_tensor_q_shape[kNumIndex3] <= 0 || input_tensor_q_shape[kNumIndex1] <= 0) {
    MS_LOG(ERROR) << "D is -1";
    return nullptr;
  }

  float scale_value = 1 / (pow(input_tensor_q_shape[kNumIndex3], 0.5));
  int64_t seq_len = input_tensor_q_shape[kNumIndex2];
  int64_t num_key_value_heads = input_tensor_k_shape[1];
  if (seq_len != 1) {
    return CreatePromptFlashAttentionCnodeForBNSD(
      func_graph, node, q, k, v, atten_mask, input_tensor_q_shape[kNumIndex1], 0, scale_value, num_key_value_heads);
  }
  MS_LOG(INFO) << "New fusion node is null";
  return nullptr;
}

const AnfNodePtr FlashAttentionFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                               const EquivPtr &equiv) const {
  MS_LOG(INFO) << "Do FlashAttention fusion.";
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    MS_LOG(ERROR) << "Function graph, node or equiv is nullptr.";
    return nullptr;
  }
  if (!utils::isa<CNodePtr>(node)) {
    MS_LOG(ERROR) << "This node is not cnode, node name: " << node->fullname_with_scope();
    return nullptr;
  }
  if (IsMarkedTrainOp(utils::cast<CNodePtr>(node))) {
    MS_LOG(ERROR) << "node is train op, can not fusion.";
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

}  // namespace mindspore::opt
