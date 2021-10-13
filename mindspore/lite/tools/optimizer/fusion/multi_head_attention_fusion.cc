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
#include "tools/optimizer/fusion/multi_head_attention_fusion.h"
#include <functional>
#include <utility>
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl/op_base.h"

namespace mindspore::opt {
namespace {
const auto &p1 = std::placeholders::_1;
const size_t kWeightShapeSize = 2;
}  // namespace

namespace {
VectorRef DefineEmbedding(const BaseRef &input, const BaseRef &weight, const BaseRef &bias) {
  auto is_matmul = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMul));
  MS_CHECK_TRUE_RET(is_matmul != nullptr, {});
  auto dense = VectorRef({is_matmul, input, weight, bias});
  auto is_reshape = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape));
  MS_CHECK_TRUE_RET(is_reshape != nullptr, {});
  auto is_var = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var != nullptr, {});
  auto reshape = VectorRef({is_reshape, dense, is_var});
  auto is_transpose = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTranspose));
  MS_CHECK_TRUE_RET(is_transpose != nullptr, {});
  auto is_param = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_param != nullptr, {});
  return VectorRef({is_transpose, reshape, is_param});
}

VectorRef DefineMask(const BaseRef &mask_input) {
  auto is_expand_dims = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimExpandDims));
  MS_CHECK_TRUE_RET(is_expand_dims != nullptr, {});
  auto is_param1 = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_param1 != nullptr, {});
  auto expand_dims = VectorRef({is_expand_dims, mask_input, is_param1});
  auto is_sub = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimSubFusion));
  MS_CHECK_TRUE_RET(is_sub != nullptr, {});
  auto is_param2 = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_param2 != nullptr, {});
  auto sub = VectorRef({is_sub, is_param2, expand_dims});
  auto is_mul = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMulFusion));
  MS_CHECK_TRUE_RET(is_mul != nullptr, {});
  auto is_param3 = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_param3 != nullptr, {});
  return VectorRef({is_mul, sub, is_param3});
}
}  // namespace

VectorRef MultiHeadAttentionFusion::DefineMPWithMaskPattern() const {
  auto q_embedding = DefineEmbedding(input_q_, weight_q_, bias_q_);
  MS_CHECK_TRUE_RET(!q_embedding.empty(), {});
  auto k_embedding = DefineEmbedding(input_k_, weight_k_, bias_k_);
  MS_CHECK_TRUE_RET(!k_embedding.empty(), {});
  auto v_embedding = DefineEmbedding(input_v_, weight_v_, bias_v_);
  MS_CHECK_TRUE_RET(!v_embedding.empty(), {});
  auto is_matmul1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMul));
  MS_CHECK_TRUE_RET(is_matmul1 != nullptr, {});
  auto q2k = VectorRef({is_matmul1, q_embedding, k_embedding});
  auto is_mul = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMulFusion));
  MS_CHECK_TRUE_RET(is_mul != nullptr, {});
  auto is_param1 = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_param1 != nullptr, {});
  auto q2k_normed = VectorRef({is_mul, q2k, is_param1});
  auto mask = DefineMask(mask_);
  MS_CHECK_TRUE_RET(!mask.empty(), {});
  auto is_add = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAddFusion));
  MS_CHECK_TRUE_RET(is_add != nullptr, {});
  auto q2k_normed_masked = VectorRef({is_add, q2k_normed, mask});
  auto is_softmax = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimSoftmax));
  MS_CHECK_TRUE_RET(is_softmax != nullptr, {});
  auto softmax = VectorRef({is_softmax, q2k_normed_masked});
  auto is_matmul2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMul));
  MS_CHECK_TRUE_RET(is_matmul2 != nullptr, {});
  auto softmax2v = VectorRef({is_matmul2, softmax, v_embedding});
  auto is_transpose = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTranspose));
  MS_CHECK_TRUE_RET(is_transpose != nullptr, {});
  auto is_param2 = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_param2 != nullptr, {});
  auto softmax2v_transposed = VectorRef({is_transpose, softmax2v, is_param2});
  auto is_reshape = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape));
  MS_CHECK_TRUE_RET(is_reshape != nullptr, {});
  auto is_var = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var != nullptr, {});
  auto softmax2v_transposed_reshaped = VectorRef({is_reshape, softmax2v_transposed, is_var});
  auto is_matmul3 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMul));
  MS_CHECK_TRUE_RET(is_matmul3 != nullptr, {});
  return VectorRef({is_matmul3, softmax2v_transposed_reshaped, weight_o_, bias_o_});
}

namespace {
VectorRef DefineDensePattern(const BaseRef &input, const BaseRef &weight, const BaseRef &bias) {
  auto is_tranpose = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTranspose));
  MS_CHECK_TRUE_RET(is_tranpose != nullptr, {});
  auto is_param1 = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_param1 != nullptr, {});
  auto transpose = VectorRef({is_tranpose, input, is_param1});
  auto is_reshape1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape));
  MS_CHECK_TRUE_RET(is_reshape1 != nullptr, {});
  auto is_param2 = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_param2 != nullptr, {});
  auto reshape1 = VectorRef({is_reshape1, transpose, is_param2});
  auto is_matmul = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMul));
  MS_CHECK_TRUE_RET(is_matmul != nullptr, {});
  auto matmul = VectorRef({is_matmul, reshape1, weight});
  auto is_reshape2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape));
  MS_CHECK_TRUE_RET(is_reshape2 != nullptr, {});
  auto is_param3 = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_param3 != nullptr, {});
  auto reshape2 = VectorRef({is_reshape2, matmul, is_param3});
  if (bias == nullptr) {
    return reshape2;
  }
  auto is_bias_add = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimBiasAdd));
  MS_CHECK_TRUE_RET(is_bias_add != nullptr, {});
  auto bias_add = VectorRef({is_bias_add, reshape2, bias});
  return bias_add;
}

VectorRef DefineProcessInputPattern(const BaseRef &input, const BaseRef &weight, const BaseRef &bias,
                                    const BaseRef &reshape_shape, bool transpose = false) {
  auto input_after_dense = DefineDensePattern(input, weight, bias);
  MS_CHECK_TRUE_RET(!input_after_dense.empty(), {});
  auto is_reshape = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape));
  MS_CHECK_TRUE_RET(is_reshape != nullptr, {});
  auto result = VectorRef({is_reshape, input_after_dense, reshape_shape});
  if (transpose) {
    auto is_transpose = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTranspose));
    MS_CHECK_TRUE_RET(is_transpose != nullptr, {});
    auto is_param = std::make_shared<CondVar>(IsParamNode);
    result = VectorRef({is_transpose, result, is_param});
  }
  return result;
}

VectorRef DefineProcessOutputPattern(const BaseRef &input, const BaseRef &weight, const BaseRef &bias) {
  auto is_transpose = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTranspose));
  MS_CHECK_TRUE_RET(is_transpose != nullptr, {});
  auto is_param1 = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_param1 != nullptr, {});
  auto transpose = VectorRef({is_transpose, input, is_param1});
  auto is_reshape = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape));
  MS_CHECK_TRUE_RET(is_reshape != nullptr, {});
  auto is_param2 = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_param2 != nullptr, {});
  auto reshape = VectorRef({is_reshape, transpose, is_param2});
  return DefineDensePattern(reshape, weight, bias);
}
}  // namespace

bool MultiHeadAttentionFusion::Init() const {
  input_q_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(input_q_ != nullptr, false);
  input_k_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(input_k_ != nullptr, false);
  input_v_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(input_v_ != nullptr, false);

  weight_q_ = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(weight_q_ != nullptr, false);
  weight_k_ = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(weight_k_ != nullptr, false);
  weight_v_ = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(weight_v_ != nullptr, false);
  weight_o_ = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(weight_o_ != nullptr, false);

  bias_q_ = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(bias_q_ != nullptr, false);
  bias_k_ = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(bias_k_ != nullptr, false);
  bias_v_ = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(bias_v_ != nullptr, false);
  bias_o_ = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(bias_o_ != nullptr, false);

  mask_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(mask_ != nullptr, false);

  reshape_k_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(reshape_k_ != nullptr, false);
  reshape_v_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(reshape_v_ != nullptr, false);
  return true;
}

VectorRef MultiHeadAttentionFusion::DefineMPWithoutMaskPattern() const {
  auto is_param1 = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_param1 != nullptr, {});
  auto query = DefineProcessInputPattern(input_q_, weight_q_, bias_q_, is_param1);
  MS_CHECK_TRUE_RET(!query.empty(), {});
  auto is_div = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimDivFusion));
  MS_CHECK_TRUE_RET(is_div != nullptr, {});
  auto is_param2 = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_param2 != nullptr, {});
  auto query_div = VectorRef({is_div, query, is_param2});

  auto key = DefineProcessInputPattern(input_k_, weight_k_, bias_k_, reshape_k_);
  MS_CHECK_TRUE_RET(!key.empty(), {});
  auto is_matmul1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMul));
  MS_CHECK_TRUE_RET(is_matmul1 != nullptr, {});
  auto query_mul_key = VectorRef({is_matmul1, query_div, key});
  auto is_softmax = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimSoftmax));
  MS_CHECK_TRUE_RET(is_softmax != nullptr, {});
  auto softmax = VectorRef({is_softmax, query_mul_key});

  auto value = DefineProcessInputPattern(input_v_, weight_v_, bias_v_, reshape_v_);
  MS_CHECK_TRUE_RET(!value.empty(), {});
  auto is_matmul2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMul));
  MS_CHECK_TRUE_RET(is_matmul2 != nullptr, {});
  auto softmax_mul_val = VectorRef({is_matmul2, softmax, value});

  return DefineProcessOutputPattern(softmax_mul_val, weight_o_, bias_o_);
}

std::unordered_map<std::string, VectorRef> MultiHeadAttentionFusion::DefinePatterns() const {
  std::unordered_map<std::string, VectorRef> patterns;
  if (!Init()) {
    MS_LOG(ERROR) << "initial member failed.";
    return patterns;
  }
  patterns[kMPAWithoutMaskPatternName] = DefineMPWithoutMaskPattern();
  patterns[kMPAWithMaskPatternName] = DefineMPWithMaskPattern();
  return patterns;
}

AnfNodePtr MultiHeadAttentionFusion::Process(const std::string &pattern_name, const mindspore::FuncGraphPtr &func_graph,
                                             const mindspore::AnfNodePtr &node,
                                             const mindspore::EquivPtr &equiv) const {
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    return nullptr;
  }
  if (pattern_name == kMPAWithoutMaskPatternName) {
    return CreateMultiHeadAttentionNode(func_graph, equiv, node->fullname_with_scope());
  } else if (pattern_name == kMPAWithMaskPatternName) {
    return CreateMaskedMultiHeadAttentionNode(func_graph, equiv, node->fullname_with_scope());
  } else {
    return nullptr;
  }
}

CNodePtr MultiHeadAttentionFusion::CreateMultiHeadAttentionNode(const FuncGraphPtr &func_graph, const EquivPtr &equiv,
                                                                const std::string &base_name) const {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(equiv != nullptr);
  auto attention_prim = BuildAttentionPrim(equiv);
  if (attention_prim == nullptr) {
    MS_LOG(ERROR) << "Build attention primitive failed.";
    return nullptr;
  }
  auto value_node = NewValueNode(attention_prim);
  MS_CHECK_TRUE_RET(value_node != nullptr, nullptr);
  auto input_q = utils::cast<AnfNodePtr>((*equiv)[input_q_]);
  auto input_k = utils::cast<AnfNodePtr>((*equiv)[input_k_]);
  auto input_v = utils::cast<AnfNodePtr>((*equiv)[input_v_]);

  auto weight_q = utils::cast<AnfNodePtr>((*equiv)[weight_q_]);
  auto weight_k = utils::cast<AnfNodePtr>((*equiv)[weight_k_]);
  auto weight_v = utils::cast<AnfNodePtr>((*equiv)[weight_v_]);
  auto weight_o = utils::cast<AnfNodePtr>((*equiv)[weight_o_]);

  auto bias_q = utils::cast<AnfNodePtr>((*equiv)[bias_q_]);
  auto bias_k = utils::cast<AnfNodePtr>((*equiv)[bias_k_]);
  auto bias_v = utils::cast<AnfNodePtr>((*equiv)[bias_v_]);
  auto bias_o = utils::cast<AnfNodePtr>((*equiv)[bias_o_]);

  std::vector<AnfNodePtr> new_node_inputs = {value_node, input_q,  input_k, input_v, weight_q, weight_k,
                                             weight_v,   weight_o, bias_q,  bias_k,  bias_v,   bias_o};
  auto new_node = func_graph->NewCNode(new_node_inputs);
  MS_CHECK_TRUE_RET(new_node != nullptr, nullptr);
  new_node->set_fullname_with_scope(base_name);
  return new_node;
}

STATUS GetIntParameterData(const ParameterPtr &param_ptr, std::vector<int> *result) {
  if (param_ptr == nullptr || !param_ptr->has_default()) {
    MS_LOG(DEBUG) << "param not have default";
    return RET_ERROR;
  }
  auto default_param = param_ptr->default_param();
  if (default_param == nullptr || !utils::isa<tensor::TensorPtr>(default_param)) {
    MS_LOG(DEBUG) << "tensor_info is not tensor::TensorPtr";
    return RET_ERROR;
  }
  auto default_param_ptr = utils::cast<tensor::TensorPtr>(default_param);
  if (default_param_ptr->data_type() != kNumberTypeInt32 && default_param_ptr->data_type() != kNumberTypeInt) {
    MS_LOG(DEBUG) << "default param is not int";
    return RET_ERROR;
  }
  auto ptr = reinterpret_cast<int *>(default_param_ptr->data_c());
  int64_t shape_size =
    std::accumulate(default_param_ptr->shape().begin(), default_param_ptr->shape().end(), 1, std::multiplies<>());
  for (int64_t i = 0; i < shape_size; i++) {
    result->emplace_back(ptr[i]);
  }
  return RET_OK;
}

std::shared_ptr<ops::Attention> MultiHeadAttentionFusion::BuildAttentionPrim(const EquivPtr &equiv) const {
  MS_ASSERT(equiv != nullptr);
  auto attention_prim = std::make_shared<ops::Attention>();
  if (attention_prim == nullptr) {
    MS_LOG(ERROR) << "Build attention primitive failed.";
    return attention_prim;
  }
  if (!utils::isa<ParameterPtr>((*equiv)[reshape_k_])) {
    MS_LOG(ERROR) << "Reshape k is not a parameter";
    return nullptr;
  }

  if (!utils::isa<ParameterPtr>((*equiv)[reshape_v_])) {
    MS_LOG(ERROR) << "Reshape k is not a parameter";
    return nullptr;
  }

  auto reshape_k = utils::cast<ParameterPtr>((*equiv)[reshape_k_]);
  std::vector<int> shape_k;
  if (RET_OK != GetIntParameterData(reshape_k, &shape_k)) {
    MS_LOG(ERROR) << "Get reshape k data failed";
    return nullptr;
  }

  auto reshape_v = utils::cast<ParameterPtr>((*equiv)[reshape_v_]);
  std::vector<int> shape_v;
  if (RET_OK != GetIntParameterData(reshape_v, &shape_v)) {
    MS_LOG(ERROR) << "Get reshape k data failed";
    return nullptr;
  }
  if (shape_k.size() < kWeightShapeSize || shape_v.size() < kWeightShapeSize ||
      shape_k.at(shape_k.size() - kWeightShapeSize) != shape_v.at(shape_v.size() - kWeightShapeSize)) {
    MS_LOG(ERROR) << "Shape k or shape v is invalid.";
    return nullptr;
  }
  return attention_prim;
}

CNodePtr MultiHeadAttentionFusion::CreateMaskedMultiHeadAttentionNode(const FuncGraphPtr &func_graph,
                                                                      const EquivPtr &equiv,
                                                                      const string &base_name) const {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(equiv != nullptr);
  auto attention_prim = std::make_shared<ops::Attention>();
  if (attention_prim == nullptr) {
    MS_LOG(ERROR) << "Build attention primitive failed.";
    return nullptr;
  }
  auto value_node = NewValueNode(attention_prim);
  MS_CHECK_TRUE_RET(value_node != nullptr, nullptr);
  auto input_q = utils::cast<AnfNodePtr>((*equiv)[input_q_]);
  auto input_k = utils::cast<AnfNodePtr>((*equiv)[input_k_]);
  auto input_v = utils::cast<AnfNodePtr>((*equiv)[input_v_]);

  auto weight_q = utils::cast<AnfNodePtr>((*equiv)[weight_q_]);
  auto weight_k = utils::cast<AnfNodePtr>((*equiv)[weight_k_]);
  auto weight_v = utils::cast<AnfNodePtr>((*equiv)[weight_v_]);
  auto weight_o = utils::cast<AnfNodePtr>((*equiv)[weight_o_]);

  auto bias_q = utils::cast<AnfNodePtr>((*equiv)[bias_q_]);
  auto bias_k = utils::cast<AnfNodePtr>((*equiv)[bias_k_]);
  auto bias_v = utils::cast<AnfNodePtr>((*equiv)[bias_v_]);
  auto bias_o = utils::cast<AnfNodePtr>((*equiv)[bias_o_]);
  auto mask = utils::cast<AnfNodePtr>((*equiv)[mask_]);

  std::vector<AnfNodePtr> new_node_inputs = {value_node, input_q, input_k, input_v, weight_q, weight_k, weight_v,
                                             weight_o,   bias_q,  bias_k,  bias_v,  bias_o,   mask};
  auto new_node = func_graph->NewCNode(new_node_inputs);
  MS_CHECK_TRUE_RET(new_node != nullptr, nullptr);
  new_node->set_fullname_with_scope(base_name);
  return new_node;
}
}  // namespace mindspore::opt
