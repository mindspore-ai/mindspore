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
#include "tools/optimizer/fusion/decoder_layer_fusion.h"
#include <functional>
#include <utility>
#include <vector>
#include <algorithm>
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl/op_base.h"
#include "ops/tuple_get_item.h"
#include "tools/common/tensor_util.h"
#include "ops/op_utils.h"

namespace mindspore::opt {
namespace {
const auto &p1 = std::placeholders::_1;
}  // namespace

bool DecoderLayerFusion::Init() const {
  hidden_stats_ = std::make_shared<Var>("input");
  MS_CHECK_TRUE_RET(hidden_stats_ != nullptr, false);
  encoder_output_ = std::make_shared<Var>("input");
  MS_CHECK_TRUE_RET(encoder_output_ != nullptr, false);
  beta1_ = std::make_shared<Var>("beta1");
  MS_CHECK_TRUE_RET(beta1_ != nullptr, false);
  gamma1_ = std::make_shared<Var>("gamma1");
  MS_CHECK_TRUE_RET(gamma1_ != nullptr, false);
  beta2_ = std::make_shared<Var>("beta2");
  MS_CHECK_TRUE_RET(beta2_ != nullptr, false);
  gamma2_ = std::make_shared<Var>("gamma2");
  MS_CHECK_TRUE_RET(gamma2_ != nullptr, false);
  beta3_ = std::make_shared<Var>("beta3");
  MS_CHECK_TRUE_RET(beta3_ != nullptr, false);
  gamma3_ = std::make_shared<Var>("gamma3");
  MS_CHECK_TRUE_RET(gamma3_ != nullptr, false);
  gamma4_ = std::make_shared<Var>("gamma4");
  MS_CHECK_TRUE_RET(gamma4_ != nullptr, false);
  beta4_ = std::make_shared<Var>("beta4");
  MS_CHECK_TRUE_RET(beta4_ != nullptr, false);
  weight_attn_qkv_ = std::make_shared<Var>("weight_attn_qkv");
  MS_CHECK_TRUE_RET(weight_attn_qkv_ != nullptr, false);
  weight_attn_q_ = std::make_shared<Var>("weight_attn_q_");
  MS_CHECK_TRUE_RET(weight_attn_q_ != nullptr, false);
  weight_attn_kv_ = std::make_shared<Var>("weight_attn_kv_");
  MS_CHECK_TRUE_RET(weight_attn_kv_ != nullptr, false);
  weight_attn_o_ = std::make_shared<CondVar>(IsParamNode, "weight_attn_o");
  MS_CHECK_TRUE_RET(weight_attn_o_ != nullptr, false);
  weight_attn_cross_o_ = std::make_shared<CondVar>(IsParamNode, "weight_attn_cross_o_");
  MS_CHECK_TRUE_RET(weight_attn_cross_o_ != nullptr, false);
  weight_m_ = std::make_shared<CondVar>(IsParamNode, "weight_m");
  MS_CHECK_TRUE_RET(weight_m_ != nullptr, false);
  weight_p_ = std::make_shared<CondVar>(IsParamNode, "weight_p");
  MS_CHECK_TRUE_RET(weight_p_ != nullptr, false);
  bias_attn_qkv_ = std::make_shared<Var>("bias_attn_qkv");
  MS_CHECK_TRUE_RET(bias_attn_qkv_ != nullptr, false);
  bias_attn_o_ = std::make_shared<CondVar>(IsParamNode, "bias_attn_o");
  MS_CHECK_TRUE_RET(bias_attn_o_ != nullptr, false);
  bias_attn_cross_qkv_ = std::make_shared<Var>("bias_attn_cross_qkv_");
  MS_CHECK_TRUE_RET(bias_attn_cross_qkv_ != nullptr, false);
  bias_attn_cross_o_ = std::make_shared<CondVar>(IsParamNode, "bias_attn_cross_o_");
  MS_CHECK_TRUE_RET(bias_attn_cross_o_ != nullptr, false);
  bias_m_ = std::make_shared<CondVar>(IsParamNode, "bias_m");
  MS_CHECK_TRUE_RET(bias_m_ != nullptr, false);
  bias_p_ = std::make_shared<CondVar>(IsParamNode, "bias_p");
  MS_CHECK_TRUE_RET(bias_p_ != nullptr, false);
  mask_ = std::make_shared<Var>("mask");
  MS_CHECK_TRUE_RET(mask_ != nullptr, false);
  cross_mask_ = std::make_shared<Var>("cross_mask_");
  MS_CHECK_TRUE_RET(cross_mask_ != nullptr, false);
  is_attention_ = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAttention), "is_attention");
  MS_CHECK_TRUE_RET(is_attention_ != nullptr, false);
  is_attention_cross_ = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAttention), "is_attention_cross");
  MS_CHECK_TRUE_RET(is_attention_cross_ != nullptr, false);
  is_layernorm1_ = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimLayerNormFusion), "layer_norm1");
  MS_CHECK_TRUE_RET(is_layernorm1_ != nullptr, false);
  is_layernorm2_ = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimLayerNormFusion), "layer_norm2");
  MS_CHECK_TRUE_RET(is_layernorm2_ != nullptr, false);
  is_layernorm3_ = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimLayerNormFusion), "layer_norm3");
  MS_CHECK_TRUE_RET(is_layernorm3_ != nullptr, false);
  position_bias_ = std::make_shared<Var>("position_bias");
  MS_CHECK_TRUE_RET(position_bias_ != nullptr, false);
  position_bias_cross_ = std::make_shared<Var>("position_bias_cross_");
  MS_CHECK_TRUE_RET(position_bias_ != nullptr, false);
  is_act_ = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimActivation), "activation");
  MS_CHECK_TRUE_RET(is_act_ != nullptr, false);
  eps1_ = std::make_shared<Var>("eps1_");
  MS_CHECK_TRUE_RET(eps1_ != nullptr, false);
  eps2_ = std::make_shared<Var>("eps2_");
  MS_CHECK_TRUE_RET(eps2_ != nullptr, false);
  eps3_ = std::make_shared<Var>("eps3_");
  MS_CHECK_TRUE_RET(eps3_ != nullptr, false);
  eps4_ = std::make_shared<Var>("eps4_");
  MS_CHECK_TRUE_RET(eps4_ != nullptr, false);
  return true;
}

VectorRef DecoderLayerFusion::getTuple(bool post_layernorm, bool layernorm_fusion = false,
                                       bool is_position_bias = false) const {
  auto is_reshape1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "reshape-decoder");
  MS_CHECK_TRUE_RET(is_reshape1 != nullptr, {});
  auto var1 = std::make_shared<Var>("var1-reshape");
  MS_CHECK_TRUE_RET(var1 != nullptr, {});
  auto reshape1 = VectorRef({is_reshape1, hidden_stats_, var1});
  VectorRef layer_norm, tuple;
  if (!layernorm_fusion) {
    return DefineLayerNorm(reshape1, gamma1_, beta1_, eps1_);
  }
  layer_norm = VectorRef({is_layernorm1_, reshape1, gamma1_, beta1_});
  auto is_tuple = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTupleGetItem), "tuple_get_itme");
  auto var_tuple = std::make_shared<Var>("var_tuple");
  tuple = VectorRef({is_tuple, layer_norm, var_tuple});
  return tuple;
}

VectorRef DecoderLayerFusion::DefineLayerNorm(VectorRef input, VarPtr gamma, VarPtr beta, VarPtr eps) const {
  auto is_sqr = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimSquare), "sqr2");
  MS_CHECK_TRUE_RET(is_sqr != nullptr, {});
  auto sqr = VectorRef({is_sqr, input});
  auto var1 = std::make_shared<Var>("var1");
  MS_CHECK_TRUE_RET(var1 != nullptr, {});
  auto is_reduce = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReduceFusion), "reduce");
  MS_CHECK_TRUE_RET(is_reduce != nullptr, {});
  auto reduce = VectorRef({is_reduce, sqr, var1});
  auto is_add = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAddFusion), "is-add");
  MS_CHECK_TRUE_RET(is_add != nullptr, {});
  auto add = VectorRef({is_add, reduce, eps});
  auto is_sqrt = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimSqrt), "sqr2");
  MS_CHECK_TRUE_RET(is_sqrt != nullptr, {});
  auto sqrt = VectorRef({is_sqrt, add});
  auto is_div = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimRealDiv), "real-div");
  MS_CHECK_TRUE_RET(is_div != nullptr, {});
  auto real_div = VectorRef({is_div, input, sqrt});
  auto is_mul = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMulFusion), "mul");
  MS_CHECK_TRUE_RET(is_mul != nullptr, {});
  auto mul = VectorRef({is_mul, real_div, gamma});
  return mul;
}

VectorRef DecoderLayerFusion::DefinePatternDecoderLayer(bool post_layernorm = true, bool layernorm_fusion = false,
                                                        bool is_position_bias = false, bool mask = true,
                                                        bool is_layer_norm = false) const {
  auto is_reshape1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "reshape-decoder");
  MS_CHECK_TRUE_RET(is_reshape1 != nullptr, {});
  auto var1 = std::make_shared<Var>("var1-reshape");
  MS_CHECK_TRUE_RET(var1 != nullptr, {});
  auto reshape1 = VectorRef({is_reshape1, hidden_stats_, var1});
  VectorRef inputs, input_cross, tuple2, tuple3, matmul2, tuple4, tuple5;
  if (is_position_bias) {
    inputs = VectorRef({is_attention_, getTuple(post_layernorm, layernorm_fusion, is_position_bias),
                        getTuple(post_layernorm, layernorm_fusion, is_position_bias),
                        getTuple(post_layernorm, layernorm_fusion, is_position_bias), weight_attn_qkv_, weight_attn_o_,
                        position_bias_});
  } else {
    inputs = VectorRef({is_attention_, getTuple(post_layernorm, layernorm_fusion, is_position_bias),
                        getTuple(post_layernorm, layernorm_fusion, is_position_bias),
                        getTuple(post_layernorm, layernorm_fusion, is_position_bias), weight_attn_qkv_, weight_attn_o_,
                        bias_attn_qkv_, bias_attn_o_});
  }
  if (mask) inputs.push_back(mask_);
  auto attention = VectorRef(inputs);
  if (is_position_bias) {
    tuple4 = attention;
  } else {
    auto is_tuple4 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTupleGetItem), "tuple_get_item4");
    auto var_tuple4 = std::make_shared<Var>("var_tuple4");
    tuple4 = VectorRef({is_tuple4, attention, var_tuple4});
  }
  auto is_add2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAddFusion), "is_add2");
  auto add2 = (post_layernorm)
                ? VectorRef({is_add2, getTuple(post_layernorm, layernorm_fusion, is_position_bias), tuple4})
                : VectorRef({is_add2, reshape1, tuple4});
  if (layernorm_fusion) {
    auto layer_norm2 = VectorRef({is_layernorm2_, add2, gamma2_, beta2_});
    auto is_tuple2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTupleGetItem), "tuple_get_item2");
    auto var_tuple2 = std::make_shared<Var>("var_tuple2");
    tuple2 = VectorRef({is_tuple2, layer_norm2, var_tuple2});
  } else {
    tuple2 = DefineLayerNorm(add2, gamma2_, beta2_, eps2_);
  }
  auto is_reshape2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "reshape-decoder2");
  MS_CHECK_TRUE_RET(is_reshape2 != nullptr, {});
  auto var2 = std::make_shared<Var>("var2");
  MS_CHECK_TRUE_RET(var2 != nullptr, {});
  auto reshape2 = VectorRef({is_reshape2, encoder_output_, var2});
  if (is_position_bias) {
    input_cross = VectorRef({is_attention_cross_, tuple2, reshape2, reshape2, weight_attn_q_, weight_attn_kv_,
                             weight_attn_cross_o_, position_bias_cross_});
  } else {
    input_cross = VectorRef({is_attention_cross_, tuple2, reshape2, reshape2, weight_attn_q_, weight_attn_kv_,
                             weight_attn_cross_o_, bias_attn_cross_qkv_, bias_attn_cross_o_});
  }
  if (mask) input_cross.push_back(cross_mask_);
  auto attention_cross = VectorRef(input_cross);
  if (is_position_bias) {
    tuple5 = attention_cross;
  } else {
    auto is_tuple5 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTupleGetItem), "tuple_get_item5");
    auto var_tuple5 = std::make_shared<Var>("var_tuple5");
    tuple5 = VectorRef({is_tuple5, attention_cross, var_tuple5});
  }
  auto is_add3 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAddFusion), "is_add3");
  MS_CHECK_TRUE_RET(is_add2 != nullptr, {});
  auto add3 = (post_layernorm) ? VectorRef({is_add3, tuple2, tuple5}) : VectorRef({is_add3, add2, tuple5});
  if (layernorm_fusion) {
    auto layer_norm3 = VectorRef({is_layernorm3_, add3, gamma3_, beta3_});
    auto is_tuple3 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTupleGetItem), "tuple_get_item3");
    auto var_tuple3 = std::make_shared<Var>("var_tuple3");
    tuple3 = VectorRef({is_tuple3, layer_norm3, var_tuple3});
  } else {
    tuple3 = DefineLayerNorm(add3, gamma3_, beta3_, eps3_);
  }
  auto is_matmul1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMulFusion), "is_matmul1");
  MS_CHECK_TRUE_RET(is_matmul1 != nullptr, {});
  auto is_matmul2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMulFusion), "is_matmul2");
  MS_CHECK_TRUE_RET(is_matmul2 != nullptr, {});
  if (!is_position_bias) {
    auto matmul1 = VectorRef({is_matmul1, tuple3, weight_m_, bias_m_});
    auto act = VectorRef({is_act_, matmul1});
    matmul2 = VectorRef({is_matmul2, act, weight_p_, bias_p_});
  } else {
    auto matmul1 = VectorRef({is_matmul1, tuple3, weight_m_});
    matmul2 = VectorRef({is_matmul2, matmul1, weight_p_});
  }
  auto is_reshape3 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "reshape-decoder3");
  MS_CHECK_TRUE_RET(is_reshape3 != nullptr, {});
  auto var3 = std::make_shared<Var>("var3");
  MS_CHECK_TRUE_RET(var3 != nullptr, {});
  auto reshape3 = VectorRef({is_reshape3, matmul2, var3});
  auto is_reshape4 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "reshape-decoder4");
  MS_CHECK_TRUE_RET(is_reshape4 != nullptr, {});
  auto var4 = std::make_shared<Var>("var4");
  MS_CHECK_TRUE_RET(var4 != nullptr, {});
  auto reshape4 = (post_layernorm) ? VectorRef({is_reshape4, tuple3, var4}) : VectorRef({is_reshape4, add3, var4});
  auto is_add4 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAddFusion), "is_add4");
  auto add4 = VectorRef({is_add4, reshape4, reshape3});
  return (is_layer_norm) ? DefineLayerNorm(add4, gamma4_, beta4_, eps4_) : add4;
}

std::unordered_map<std::string, VectorRef> DecoderLayerFusion::DefinePatterns() const {
  std::unordered_map<std::string, VectorRef> patterns;
  if (!Init()) {
    MS_LOG(ERROR) << "initial member failed.";
    return patterns;
  }
  patterns[kPatternDecoderLayerNormT5Pre] = DefinePatternDecoderLayer(false, false, true, true, true);
  patterns[kPatternDecoderLayerPre] = DefinePatternDecoderLayer(false, true, false, true);
  patterns[kPatternDecoderLayerPost] = DefinePatternDecoderLayer(true, true, false, true);
  patterns[kPatternDecoderLayerNormPre] = DefinePatternDecoderLayer(false, false, false, true);
  patterns[kPatternDecoderLayerNormPost] = DefinePatternDecoderLayer(true, false, false, true);
  patterns[kPatternDecoderT5Pre] = DefinePatternDecoderLayer(false, false, true, true);
  patterns[kPatternDecoderT5Post] = DefinePatternDecoderLayer(true, false, true, true);
  return patterns;
}

AnfNodePtr DecoderLayerFusion::Process(const std::string &pattern_name, const mindspore::FuncGraphPtr &func_graph,
                                       const mindspore::AnfNodePtr &node, const mindspore::EquivPtr &equiv) const {
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    return nullptr;
  }
  if (pattern_name == kPatternDecoderT5Pre || pattern_name == kPatternDecoderT5Post ||
      pattern_name == kPatternDecoderLayerNormT5Pre) {
    is_position_bias_ = true;
  }
  is_layernorm_ = false;
  if (pattern_name == kPatternDecoderLayerNormT5Pre) {
    is_layernorm_ = true;
  }
  if (pattern_name == kPatternDecoderLayerPre || pattern_name == kPatternDecoderLayerPost) {
    is_layernorm_fusion_ = true;
  }
  bool mask = true;
  bool post_layernorm = false;
  if (pattern_name == kPatternDecoderLayerPost || pattern_name == kPatternDecoderT5Post ||
      pattern_name == kPatternDecoderLayerNormPost) {
    post_layernorm = true;
  }
  return CreateMaskedDecoderLayerFusionNode(func_graph, equiv, node, post_layernorm, mask);
}  // namespace mindspore::opt

bool DecoderLayerFusion::IsActGELU(const FuncGraphPtr &func_graph, const EquivPtr &equiv) const {
  auto act_input = GetAttribute(func_graph, equiv, is_act_);
  MS_ASSERT(act_input != nullptr);
  auto act_primitive = ops::GetOperator<ops::Activation>(act_input);
  MS_CHECK_TRUE_RET(act_primitive != nullptr, false);
  auto act_primitive_c = act_primitive->GetPrim();
  if (act_primitive_c->GetAttr(ops::kActivationType) == nullptr ||
      act_primitive->get_activation_type() != mindspore::GELU) {
    return false;
  }
  return true;
}

AnfNodePtr DecoderLayerFusion::GetAttribute(const FuncGraphPtr &func_graph, const EquivPtr &equiv,
                                            VarPtr node_name) const {
  if ((*equiv)[node_name] == nullptr || !utils::isa<AnfNodePtr>((*equiv)[node_name])) {
    MS_LOG(ERROR) << node_name << "is not AnfNodePtr";
    return nullptr;
  }
  AnfNodePtr node = utils::cast<AnfNodePtr>((*equiv)[node_name]);
  MS_ASSERT(node != nullptr);
  if (node == nullptr || !utils::isa<CNodePtr>(node)) {
    auto manager = func_graph->manager();
    if (manager == nullptr) {
      return nullptr;
    }
    auto users = manager->node_users();
    auto it = users.find(node);
    if (it != users.end()) {
      node = it->second.front().first;
    }
    if (node == nullptr || !utils::isa<CNodePtr>(node)) {
      return nullptr;
    }
  }
  auto cnode = utils::cast<CNodePtr>(node);
  MS_ASSERT(cnode != nullptr);
  auto input = cnode->input(0);
  return input;
}

STATUS DecoderLayerFusion::GetEps(const EquivPtr &equiv, VarPtr node_name, float *eps) const {
  if ((*equiv)[node_name] == nullptr || !utils::isa<AnfNodePtr>((*equiv)[node_name])) {
    MS_LOG(ERROR) << node_name << " is not anfnodeptr";
    return RET_ERROR;
  }
  AnfNodePtr node = utils::cast<AnfNodePtr>((*equiv)[node_name]);
  MS_ASSERT(node != nullptr);
  if (utils::isa<ValueNodePtr>(node)) {
    auto value_ptr_node = utils::cast<ValueNodePtr>(node);
    auto value_node = utils::cast<ValuePtr>(value_ptr_node->value());
    if (value_node->isa<tensor::Tensor>()) {
      auto tensor = value_node->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor);
      *eps = *reinterpret_cast<float *>(tensor->data().data());
      return RET_OK;
    }
  }
  return RET_ERROR;
}

STATUS DecoderLayerFusion::CheckPattern(const FuncGraphPtr &func_graph, const EquivPtr &equiv, int *head_num,
                                        int *head_size, float *eps1, float *eps2, float *eps3, float *eps4,
                                        bool *is_position_bias1, bool *is_position_bias2, float *scale1,
                                        float *scale2) const {
  auto attn_input = GetAttribute(func_graph, equiv, is_attention_);
  MS_ASSERT(attn_input != nullptr);
  auto attn_prim = ops::GetOperator<ops::Attention>(attn_input);
  if (attn_prim->GetAttr(ops::kNumHeads) != nullptr) *head_num = attn_prim->get_head_num();
  if (attn_prim->GetAttr(ops::kSizePerHead) != nullptr) *head_size = attn_prim->get_head_size();
  if (attn_prim->GetAttr(ops::kPositionBias1) != nullptr) *is_position_bias1 = attn_prim->get_position_bias();
  if (attn_prim->GetAttr(ops::kScale) != nullptr) *scale1 = attn_prim->get_scale();
  auto attn_cross_input = GetAttribute(func_graph, equiv, is_attention_cross_);
  MS_ASSERT(attn_cross_input != nullptr);
  auto attn_cross_prim = ops::GetOperator<ops::Attention>(attn_cross_input);
  if (attn_cross_prim->GetAttr(ops::kPositionBias1) != nullptr)
    *is_position_bias2 = attn_cross_prim->get_position_bias();
  if (attn_cross_prim->GetAttr(ops::kScale) != nullptr) *scale2 = attn_cross_prim->get_scale();
  if (is_layernorm_fusion_) {
    auto layrn1_input = GetAttribute(func_graph, equiv, is_layernorm1_);
    auto layrn1_prim = ops::GetOperator<ops::LayerNormFusion>(layrn1_input);
    if (layrn1_prim->GetAttr(ops::kEpsilon) != nullptr) *eps1 = layrn1_prim->get_epsilon();
    auto layrn2_input = GetAttribute(func_graph, equiv, is_layernorm2_);
    auto layrn2_prim = ops::GetOperator<ops::LayerNormFusion>(layrn2_input);
    if (layrn2_prim->GetAttr(ops::kEpsilon) != nullptr) *eps2 = layrn2_prim->get_epsilon();
    auto layrn3_input = GetAttribute(func_graph, equiv, is_layernorm3_);
    auto layrn3_prim = ops::GetOperator<ops::LayerNormFusion>(layrn3_input);
    if (layrn3_prim->GetAttr(ops::kEpsilon) != nullptr) *eps3 = layrn3_prim->get_epsilon();
  } else {
    if (GetEps(equiv, eps1_, eps1) != RET_OK) {
      MS_LOG(ERROR) << "not found eps1";
      return RET_ERROR;
    }
    if (GetEps(equiv, eps2_, eps2) != RET_OK) {
      MS_LOG(ERROR) << "not found eps2";
      return RET_ERROR;
    }
    if (GetEps(equiv, eps3_, eps3) != RET_OK) {
      MS_LOG(ERROR) << "not found eps3";
      return RET_ERROR;
    }
    if (is_layernorm_) {
      if (GetEps(equiv, eps4_, eps4) != RET_OK) {
        MS_LOG(ERROR) << "not found eps4";
        return RET_ERROR;
      }
    }
  }
  if (!is_position_bias_) {
    if (!IsActGELU(func_graph, equiv)) return RET_ERROR;
    act_type_ = ActType::ActType_Gelu;
  } else {
    act_type_ = ActType::ActType_Relu;
  }
  return RET_OK;
}

std::shared_ptr<ops::DecoderLayer> DecoderLayerFusion::CreatePrim(const FuncGraphPtr &func_graph, const EquivPtr &equiv,
                                                                  bool post_layernorm, int64_t ffn_hidden_size) const {
  auto decoder_layer_prim = std::make_shared<ops::DecoderLayer>();
  if (decoder_layer_prim == nullptr) {
    MS_LOG(ERROR) << "Build decoder layer primitive failed.";
    return nullptr;
  }
  int head_num = 0;
  int head_size = 0;
  float eps1 = 1e-6;
  float eps2 = 1e-6;
  float eps3 = 1e-6;
  float eps4 = 1e-6;
  bool is_position_bias1 = false;
  bool is_position_bias2 = false;
  float scale1 = 1.0f;
  float scale2 = 1.0f;
  if (CheckPattern(func_graph, equiv, &head_num, &head_size, &eps1, &eps2, &eps3, &eps4, &is_position_bias1,
                   &is_position_bias2, &scale1, &scale2)) {
    return nullptr;
  }
  decoder_layer_prim->Init(head_num, head_size, eps1, eps2, eps3, eps4, ffn_hidden_size, is_position_bias1,
                           is_position_bias2, post_layernorm, scale1, scale2, act_type_, is_layernorm_);
  return decoder_layer_prim;
}

CNodePtr DecoderLayerFusion::CreateMaskedDecoderLayerFusionNode(const FuncGraphPtr &func_graph, const EquivPtr &equiv,
                                                                const AnfNodePtr &node, bool post_layernorm = true,
                                                                bool mask = true) const {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(equiv != nullptr);
  MS_ASSERT(node != nullptr);
  auto input = utils::cast<AnfNodePtr>((*equiv)[hidden_stats_]);
  MS_ASSERT(input != nullptr);
  auto encoder_output = utils::cast<AnfNodePtr>((*equiv)[encoder_output_]);
  MS_ASSERT(encoder_output != nullptr);
  AnfNodePtr position_bias, input_mask, bias_attn_o, bias_attn_qkv, beta1, beta2, bias_m, bias_p, beta3,
    bias_attn_cross_qkv, bias_attn_cross_o, position_bias_cross, gamma4, beta4;
  auto weight_qkv = utils::cast<AnfNodePtr>((*equiv)[weight_attn_qkv_]);
  auto weight_attn_o = utils::cast<AnfNodePtr>((*equiv)[weight_attn_o_]);
  auto weight_attn_q = utils::cast<AnfNodePtr>((*equiv)[weight_attn_q_]);
  auto weight_attn_kv = utils::cast<AnfNodePtr>((*equiv)[weight_attn_kv_]);
  auto weight_attn_cross_o = utils::cast<AnfNodePtr>((*equiv)[weight_attn_cross_o_]);
  auto weight_m = utils::cast<AnfNodePtr>((*equiv)[weight_m_]);
  auto weight_p = utils::cast<AnfNodePtr>((*equiv)[weight_p_]);
  if (is_position_bias_) {
    position_bias = utils::cast<AnfNodePtr>((*equiv)[position_bias_]);
    position_bias_cross = utils::cast<AnfNodePtr>((*equiv)[position_bias_cross_]);
  } else {
    bias_attn_o = utils::cast<AnfNodePtr>((*equiv)[bias_attn_o_]);
    bias_attn_qkv = utils::cast<AnfNodePtr>((*equiv)[bias_attn_qkv_]);
    bias_attn_cross_qkv = utils::cast<AnfNodePtr>((*equiv)[bias_attn_cross_qkv_]);
    bias_attn_cross_o = utils::cast<AnfNodePtr>((*equiv)[bias_attn_cross_o_]);
    bias_m = utils::cast<AnfNodePtr>((*equiv)[bias_m_]);
    bias_p = utils::cast<AnfNodePtr>((*equiv)[bias_p_]);
    beta1 = utils::cast<AnfNodePtr>((*equiv)[beta1_]);
    beta2 = utils::cast<AnfNodePtr>((*equiv)[beta2_]);
    beta3 = utils::cast<AnfNodePtr>((*equiv)[beta3_]);
    if (is_layernorm_) beta4 = utils::cast<AnfNodePtr>((*equiv)[beta4_]);
  }
  auto gamma1 = utils::cast<AnfNodePtr>((*equiv)[gamma1_]);
  auto gamma2 = utils::cast<AnfNodePtr>((*equiv)[gamma2_]);
  auto gamma3 = utils::cast<AnfNodePtr>((*equiv)[gamma3_]);
  if (is_layernorm_) gamma4 = utils::cast<AnfNodePtr>((*equiv)[gamma4_]);

  input_mask = mask ? utils::cast<AnfNodePtr>((*equiv)[mask_]) : nullptr;
  auto cross_mask = utils::cast<AnfNodePtr>((*equiv)[cross_mask_]);
  auto base_shape_ptr = weight_m->Shape();
  MS_EXCEPTION_IF_NULL(base_shape_ptr);
  auto input_shape_ptr = base_shape_ptr->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(input_shape_ptr);
  auto input_shape = input_shape_ptr->shape();
  MS_ASSERT(input_shape != nullptr);
  int ffn_hidden_size = (int64_t)input_shape[1];
  auto decoder_layer_prim = CreatePrim(func_graph, equiv, post_layernorm, ffn_hidden_size);
  MS_CHECK_TRUE_RET(decoder_layer_prim != nullptr, nullptr);
  auto decoder_layer_prim_c = decoder_layer_prim->GetPrim();
  MS_CHECK_TRUE_RET(decoder_layer_prim_c != nullptr, nullptr);
  auto value_node = NewValueNode(decoder_layer_prim_c);
  MS_CHECK_TRUE_RET(value_node != nullptr, nullptr);
  std::vector<AnfNodePtr> new_node_inputs = {value_node, input, gamma1};
  if (is_position_bias_) {
    new_node_inputs.insert(new_node_inputs.end(), {weight_qkv});
    if (mask) new_node_inputs.push_back(input_mask);
    new_node_inputs.insert(new_node_inputs.end(),
                           {position_bias, weight_attn_o, gamma2, encoder_output, weight_attn_q, weight_attn_kv});
    if (mask) new_node_inputs.push_back(cross_mask);
    new_node_inputs.insert(new_node_inputs.end(),
                           {position_bias_cross, weight_attn_cross_o, gamma3, weight_m, weight_p});
    if (is_layernorm_) new_node_inputs.push_back(gamma4);
  } else {
    new_node_inputs.insert(new_node_inputs.end(), {beta1, weight_qkv, bias_attn_qkv});
    if (mask) new_node_inputs.push_back(input_mask);
    new_node_inputs.insert(new_node_inputs.end(), {weight_attn_o, bias_attn_o, gamma2, beta2, encoder_output,
                                                   weight_attn_q, weight_attn_kv, bias_attn_cross_qkv});
    if (mask) new_node_inputs.push_back(cross_mask);
    new_node_inputs.insert(new_node_inputs.end(),
                           {weight_attn_cross_o, bias_attn_cross_o, gamma3, beta3, weight_m, bias_m, weight_p, bias_p});
    if (is_layernorm_) new_node_inputs.insert(new_node_inputs.end(), {gamma4, beta4});
  }
  auto new_node = func_graph->NewCNode(new_node_inputs);
  MS_CHECK_TRUE_RET(new_node != nullptr, nullptr);
  auto old_node = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(old_node->abstract() != nullptr, nullptr);
  new_node->set_abstract(old_node->abstract()->Clone());
  new_node->set_fullname_with_scope(node->fullname_with_scope() + "/decoder_layer");
  return new_node;
}
}  // namespace mindspore::opt
