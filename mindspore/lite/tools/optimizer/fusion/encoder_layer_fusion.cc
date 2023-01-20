/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "tools/optimizer/fusion/encoder_layer_fusion.h"
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

bool EncoderLayerFusion::Init() const {
  input_ = std::make_shared<Var>("input");
  MS_CHECK_TRUE_RET(input_ != nullptr, false);
  beta1_ = std::make_shared<Var>("beta1");
  MS_CHECK_TRUE_RET(beta1_ != nullptr, false);
  gamma1_ = std::make_shared<Var>("gamma1");
  MS_CHECK_TRUE_RET(gamma1_ != nullptr, false);
  beta2_ = std::make_shared<Var>("beta2");
  MS_CHECK_TRUE_RET(beta2_ != nullptr, false);
  gamma2_ = std::make_shared<Var>("gamma2");
  MS_CHECK_TRUE_RET(gamma2_ != nullptr, false);
  weight_attn_qkv_ = std::make_shared<Var>("weight_attn_qkv");
  MS_CHECK_TRUE_RET(weight_attn_qkv_ != nullptr, false);
  weight_attn_o_ = std::make_shared<CondVar>(IsParamNode, "weight_attn_o");
  MS_CHECK_TRUE_RET(weight_attn_o_ != nullptr, false);
  weight_m_ = std::make_shared<CondVar>(IsParamNode, "weight_m");
  MS_CHECK_TRUE_RET(weight_m_ != nullptr, false);
  weight_p_ = std::make_shared<CondVar>(IsParamNode, "weight_p");
  MS_CHECK_TRUE_RET(weight_p_ != nullptr, false);
  bias_attn_qkv_ = std::make_shared<Var>("bias_attn_qkv");
  MS_CHECK_TRUE_RET(bias_attn_qkv_ != nullptr, false);
  bias_attn_o_ = std::make_shared<CondVar>(IsParamNode, "bias_attn_o");
  MS_CHECK_TRUE_RET(bias_attn_o_ != nullptr, false);
  bias_m_ = std::make_shared<CondVar>(IsParamNode, "bias_m");
  MS_CHECK_TRUE_RET(bias_m_ != nullptr, false);
  bias_p_ = std::make_shared<CondVar>(IsParamNode, "bias_p");
  MS_CHECK_TRUE_RET(bias_p_ != nullptr, false);
  mask_ = std::make_shared<Var>("mask");
  MS_CHECK_TRUE_RET(mask_ != nullptr, false);
  is_attention_ = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAttention), "is_attention");
  MS_CHECK_TRUE_RET(is_attention_ != nullptr, false);
  is_layernorm1_ = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimLayerNormFusion), "layer_norm1");
  MS_CHECK_TRUE_RET(is_layernorm1_ != nullptr, false);
  is_layernorm2_ = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimLayerNormFusion), "layer_norm2");
  MS_CHECK_TRUE_RET(is_layernorm2_ != nullptr, false);
  position_bias_ = std::make_shared<Var>("position_bias");
  MS_CHECK_TRUE_RET(is_layernorm2_ != nullptr, false);
  is_act_ = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimActivation), "activation");
  MS_CHECK_TRUE_RET(is_act_ != nullptr, {});
  return true;
}

VectorRef EncoderLayerFusion::getTuple(bool post_layernorm, bool layernorm_fusion = false,
                                       bool is_position_bias = false) const {
  auto is_reshape1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "reshape-encoder");
  MS_CHECK_TRUE_RET(is_reshape1 != nullptr, {});
  auto var1 = std::make_shared<Var>("var1-reshape");
  MS_CHECK_TRUE_RET(var1 != nullptr, {});
  auto reshape1 = VectorRef({is_reshape1, input_, var1});
  if (post_layernorm) {
    return reshape1;
  }
  if (layernorm_fusion) {
    return DefineLayerNorm(is_position_bias, reshape1, gamma1_, beta1_);
  }
  auto layer_norm = VectorRef({is_layernorm1_, reshape1, gamma1_, beta1_});
  auto is_tuple = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTupleGetItem), "tuple_get_itme");
  auto var_tuple = std::make_shared<Var>("var_tuple");
  auto tuple = VectorRef({is_tuple, layer_norm, var_tuple});
  return tuple;
}

VectorRef EncoderLayerFusion::DefineLayerNorm(bool is_position_bias, VectorRef input, VarPtr gamma, VarPtr beta) const {
  auto var1 = std::make_shared<Var>("var1");
  MS_CHECK_TRUE_RET(var1 != nullptr, {});
  auto is_reduce = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReduceFusion), "reduce");
  MS_CHECK_TRUE_RET(is_reduce != nullptr, {});
  auto reduce1 = VectorRef({is_reduce, input, var1});
  auto is_sub = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimSubFusion), "sub-f");
  MS_CHECK_TRUE_RET(is_sub != nullptr, {});
  auto sub = VectorRef({is_sub, input, reduce1});
  auto is_sqr = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimSquare), "sqr");
  MS_CHECK_TRUE_RET(is_sqr != nullptr, {});
  auto sqr = (is_position_bias) ? VectorRef({is_sqr, input}) : VectorRef({is_sqr, sub});
  auto var2 = std::make_shared<Var>("var2");
  MS_CHECK_TRUE_RET(var2 != nullptr, {});
  auto is_reduce2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReduceFusion), "reduce2");
  MS_CHECK_TRUE_RET(is_reduce2 != nullptr, {});
  auto reduce2 = VectorRef({is_reduce2, sqr, var2});
  auto var3 = std::make_shared<Var>("var3");
  MS_CHECK_TRUE_RET(var3 != nullptr, {});
  auto is_add = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAddFusion), "is-add");
  MS_CHECK_TRUE_RET(is_add != nullptr, {});
  auto add = VectorRef({is_add, reduce2, var3});
  auto is_sqr2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimSqrt), "sqr2");
  MS_CHECK_TRUE_RET(is_sqr2 != nullptr, {});
  auto sqr2 = VectorRef({is_sqr2, add});
  auto is_div = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimRealDiv), "real-div");
  MS_CHECK_TRUE_RET(is_div != nullptr, {});
  if (is_position_bias) {
    auto real_div = VectorRef({is_div, input, sqr2});
    auto is_mul = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMulFusion), "mul");
    MS_CHECK_TRUE_RET(is_mul != nullptr, {});
    auto mul = VectorRef({is_mul, real_div, gamma});
    return mul;
  } else {
    auto real_div = VectorRef({is_div, sub, sqr2});
    auto is_scale = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimScaleFusion), "scale");
    MS_CHECK_TRUE_RET(is_scale != nullptr, {});
    auto scale = VectorRef({is_scale, real_div, gamma, beta});
    return scale;
  }
}

VectorRef EncoderLayerFusion::DefinePatternEncoderLayer(bool post_layernorm = true, bool layernorm_fusion = false,
                                                        bool is_position_bias = false) const {
  VectorRef attention, tuple, tuple2, tuple3, reshape2, matmul1;
  auto is_reshape1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "reshape-encoder");
  MS_CHECK_TRUE_RET(is_reshape1 != nullptr, {});
  auto var1 = std::make_shared<Var>("var1");
  MS_CHECK_TRUE_RET(var1 != nullptr, {});
  auto reshape1 = VectorRef({is_reshape1, input_, var1});
  if (!is_position_bias) {
    attention = VectorRef({is_attention_, getTuple(post_layernorm, layernorm_fusion, is_position_bias),
                           getTuple(post_layernorm, layernorm_fusion, is_position_bias),
                           getTuple(post_layernorm, layernorm_fusion, is_position_bias), weight_attn_qkv_,
                           weight_attn_o_, bias_attn_qkv_, bias_attn_o_, mask_});
  } else {
    attention = VectorRef({is_attention_, getTuple(post_layernorm, layernorm_fusion, is_position_bias),
                           getTuple(post_layernorm, layernorm_fusion, is_position_bias),
                           getTuple(post_layernorm, layernorm_fusion, is_position_bias), weight_attn_qkv_,
                           weight_attn_o_, position_bias_, mask_});
  }
  if (!is_position_bias) {
    auto is_tuple = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTupleGetItem), "tuple_get_itme");
    auto var_tuple = std::make_shared<Var>("var_tuple");
    tuple = VectorRef({is_tuple, attention, var_tuple});
  } else {
    tuple = attention;
  }
  auto is_add = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAddFusion), "is_add");
  auto add = VectorRef({is_add, reshape1, tuple});
  if (layernorm_fusion) {
    tuple2 = DefineLayerNorm(is_position_bias, add, gamma2_, beta2_);
  } else {
    auto layer_norm2 = VectorRef({is_layernorm2_, add, gamma2_, beta2_});
    auto is_tuple2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTupleGetItem), "tuple_get_item2");
    auto var_tuple2 = std::make_shared<Var>("var_tuple2");
    tuple2 = VectorRef({is_tuple2, layer_norm2, var_tuple2});
  }
  auto is_reshape2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "reshape-encoder2");
  MS_CHECK_TRUE_RET(is_reshape2 != nullptr, {});
  auto var2 = std::make_shared<Var>("var2");
  MS_CHECK_TRUE_RET(var2 != nullptr, {});
  auto is_matmul1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMulFusion), "is_matmul1");
  MS_CHECK_TRUE_RET(is_matmul1 != nullptr, {});
  if (is_position_bias) {
    reshape2 = VectorRef({is_reshape2, add, var2});
    matmul1 = VectorRef({is_matmul1, tuple2, weight_m_});
  } else if (post_layernorm || layernorm_fusion) {
    reshape2 = VectorRef({is_reshape2, tuple2, var2});
    matmul1 = VectorRef({is_matmul1, tuple2, weight_m_, bias_m_});
  } else {
    reshape2 = VectorRef({is_reshape2, add, var2});
    matmul1 = VectorRef({is_matmul1, tuple2, weight_m_, bias_m_});
  }
  auto act = VectorRef({is_act_, matmul1});
  auto is_matmul2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMulFusion), "is_matmul2");
  MS_CHECK_TRUE_RET(is_matmul2 != nullptr, {});
  auto matmul2 =
    (is_position_bias) ? VectorRef({is_matmul2, matmul1, weight_p_}) : VectorRef({is_matmul2, act, weight_p_, bias_p_});
  auto is_reshape3 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "reshape-encoder3");
  MS_CHECK_TRUE_RET(is_reshape3 != nullptr, {});
  auto var3 = std::make_shared<Var>("var3");
  MS_CHECK_TRUE_RET(var3 != nullptr, {});
  auto reshape3 = VectorRef({is_reshape3, matmul2, var3});
  auto is_add3 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAddFusion), "is_add3");
  auto add3 = VectorRef({is_add3, reshape2, reshape3});
  if (!post_layernorm || layernorm_fusion) {
    return add3;
  }
  auto is_reshape4 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "reshape-encoder");
  MS_CHECK_TRUE_RET(is_reshape4 != nullptr, {});
  auto var4 = std::make_shared<Var>("var4");
  MS_CHECK_TRUE_RET(var4 != nullptr, {});
  auto reshape4 = VectorRef({is_reshape4, add3, var4});
  if (layernorm_fusion) {
    tuple3 = DefineLayerNorm(is_position_bias, reshape4, gamma1_, beta1_);
  } else {
    auto layer_norm = VectorRef({is_layernorm1_, reshape4, gamma1_, beta1_});
    auto is_tuple3 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTupleGetItem), "tuple_get_item3");
    auto var_tuple3 = std::make_shared<Var>("var_tuple3");
    tuple3 = VectorRef({is_tuple3, layer_norm, var_tuple3});
  }
  auto is_reshape5 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "reshape-encoder");
  MS_CHECK_TRUE_RET(is_reshape5 != nullptr, {});
  auto var5 = std::make_shared<Var>("var5");
  MS_CHECK_TRUE_RET(var5 != nullptr, {});
  auto reshape5 = VectorRef({is_reshape5, tuple3, var5});
  return reshape5;
}

std::unordered_map<std::string, VectorRef> EncoderLayerFusion::DefinePatterns() const {
  std::unordered_map<std::string, VectorRef> patterns;
  if (!Init()) {
    MS_LOG(ERROR) << "initial member failed.";
    return patterns;
  }
  patterns[kPatternEncoderLayerPre] = DefinePatternEncoderLayer(false);
  patterns[kPatternEncoderLayerPost] = DefinePatternEncoderLayer(true);
  patterns[kPatternEncoderLayerPostNorm] = DefinePatternEncoderLayer(true, true);
  patterns[kPatternEncoderLayerPreNorm] = DefinePatternEncoderLayer(false, true);
  patterns[kPatternEncoderLayerT5] = DefinePatternEncoderLayer(false, true, true);
  return patterns;
}

AnfNodePtr EncoderLayerFusion::Process(const std::string &pattern_name, const mindspore::FuncGraphPtr &func_graph,
                                       const mindspore::AnfNodePtr &node, const mindspore::EquivPtr &equiv) const {
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    return nullptr;
  }
  if (pattern_name == kPatternEncoderLayerPost || pattern_name == kPatternEncoderLayerPostNorm) {
    return CreateMaskedEncoderLayerFusionNode(func_graph, equiv, node, true);
  } else if (pattern_name == kPatternEncoderLayerPre || pattern_name == kPatternEncoderLayerPreNorm) {
    return CreateMaskedEncoderLayerFusionNode(func_graph, equiv, node, false);
  } else if (pattern_name == kPatternEncoderLayerT5) {
    is_position_bias_ = true;
    return CreateMaskedEncoderLayerFusionNode(func_graph, equiv, node, false);
  }
  return nullptr;
}

bool EncoderLayerFusion::IsActGELU(const FuncGraphPtr &func_graph, const EquivPtr &equiv,
                                   const VarPtr &input_prim) const {
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

AnfNodePtr EncoderLayerFusion::GetAttribute(const FuncGraphPtr &func_graph, const EquivPtr &equiv,
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
STATUS EncoderLayerFusion::CheckPattern(const FuncGraphPtr &func_graph, const EquivPtr &equiv, int *head_num,
                                        int *head_size, float *eps1, float *eps2) const {
  auto attn_input = GetAttribute(func_graph, equiv, is_attention_);
  MS_ASSERT(attn_input != nullptr);
  auto attn_prim = ops::GetOperator<ops::Attention>(attn_input);
  if (attn_prim->GetAttr(ops::kEncoderLayerNumHeads) != nullptr) {
    *head_num = attn_prim->get_head_num();
  }
  if (attn_prim->GetAttr(ops::kAttentionSizePerHead) != nullptr) {
    *head_size = attn_prim->get_head_size();
  }
  if (attn_prim->GetAttr(ops::kPositionBias) != nullptr) {
    is_position_bias_ = attn_prim->get_position_bias();
  }
  auto layrn1_input = GetAttribute(func_graph, equiv, is_layernorm1_);
  auto layrn1_prim = ops::GetOperator<ops::LayerNormFusion>(layrn1_input);
  if (layrn1_prim->GetAttr(ops::kEpsilon) != nullptr) {
    *eps1 = layrn1_prim->get_epsilon();
  }
  auto layrn2_input = GetAttribute(func_graph, equiv, is_layernorm2_);
  auto layrn2_prim = ops::GetOperator<ops::LayerNormFusion>(layrn2_input);
  if (layrn2_prim->GetAttr(ops::kEpsilon) != nullptr) {
    *eps2 = layrn2_prim->get_epsilon();
  }
  if (!IsActGELU(func_graph, equiv, is_act_)) {
    return false;
  }
  return RET_OK;
}

std::shared_ptr<ops::EncoderLayer> EncoderLayerFusion::CreatePrim(const FuncGraphPtr &func_graph, const EquivPtr &equiv,
                                                                  bool post_layernorm, int64_t ffn_hidden_size) const {
  auto encoder_layer_prim = std::make_shared<ops::EncoderLayer>();
  if (encoder_layer_prim == nullptr) {
    MS_LOG(ERROR) << "Build enoder layer primitive failed.";
    return nullptr;
  }
  int head_num = 0;
  int head_size = 0;
  float eps1 = 1e-6;
  float eps2 = 1e-6;
  if (CheckPattern(func_graph, equiv, &head_num, &head_size, &eps1, &eps2)) {
    return nullptr;
  }
  encoder_layer_prim->Init(head_num, head_size, eps1, eps2, ffn_hidden_size, is_position_bias_, post_layernorm);
  return encoder_layer_prim;
}

CNodePtr EncoderLayerFusion::CreateMaskedEncoderLayerFusionNode(const FuncGraphPtr &func_graph, const EquivPtr &equiv,
                                                                const AnfNodePtr &node,
                                                                bool post_layernorm = true) const {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(equiv != nullptr);
  MS_ASSERT(node != nullptr);
  auto input = utils::cast<AnfNodePtr>((*equiv)[input_]);
  AnfNodePtr position_bias, input_mask, bias_attn_o, bias_attn_qkv, beta1, beta2, bias_m, bias_p;
  auto weight_qkv = utils::cast<AnfNodePtr>((*equiv)[weight_attn_qkv_]);
  auto weight_attn_o = utils::cast<AnfNodePtr>((*equiv)[weight_attn_o_]);
  auto weight_m = utils::cast<AnfNodePtr>((*equiv)[weight_m_]);
  auto weight_p = utils::cast<AnfNodePtr>((*equiv)[weight_p_]);
  if (!is_position_bias_) {
    bias_attn_qkv = utils::cast<AnfNodePtr>((*equiv)[bias_attn_qkv_]);
    bias_attn_o = utils::cast<AnfNodePtr>((*equiv)[bias_attn_o_]);
    bias_m = utils::cast<AnfNodePtr>((*equiv)[bias_m_]);
    bias_p = utils::cast<AnfNodePtr>((*equiv)[bias_p_]);
    beta1 = utils::cast<AnfNodePtr>((*equiv)[beta1_]);
    beta2 = utils::cast<AnfNodePtr>((*equiv)[beta2_]);
  }
  auto gamma1 = utils::cast<AnfNodePtr>((*equiv)[gamma1_]);
  auto gamma2 = utils::cast<AnfNodePtr>((*equiv)[gamma2_]);
  if (mask_) {
    input_mask = utils::cast<AnfNodePtr>((*equiv)[mask_]);
  }
  auto base_shape_ptr = weight_m->Shape();
  MS_EXCEPTION_IF_NULL(base_shape_ptr);
  auto input_shape_ptr = base_shape_ptr->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(input_shape_ptr);
  auto input_shape = input_shape_ptr->shape();
  MS_ASSERT(input_shape != nullptr);
  int ffn_hidden_size = (int64_t)input_shape[1];
  auto encoder_layer_prim = CreatePrim(func_graph, equiv, post_layernorm, ffn_hidden_size);
  MS_CHECK_TRUE_RET(encoder_layer_prim != nullptr, nullptr);
  auto encoder_layer_prim_c = encoder_layer_prim->GetPrim();
  MS_CHECK_TRUE_RET(encoder_layer_prim_c != nullptr, nullptr);
  auto value_node = NewValueNode(encoder_layer_prim_c);
  MS_CHECK_TRUE_RET(value_node != nullptr, nullptr);
  std::vector<AnfNodePtr> new_node_inputs;
  ParameterPtr c_bias_m_param, c_weight_p_param, c_bias_p_param, c_weight_m_param;
  if (is_position_bias_) {
    position_bias = utils::cast<AnfNodePtr>((*equiv)[position_bias_]);
    if (!post_layernorm)
      new_node_inputs = {value_node,    input,  gamma1,   weight_qkv, input_mask,
                         weight_attn_o, gamma2, weight_m, weight_p,   position_bias};
    else
      new_node_inputs = {value_node, input,    weight_qkv, input_mask, weight_attn_o,
                         gamma1,     weight_m, weight_p,   gamma2,     position_bias};
  } else {
    if (!post_layernorm) {
      new_node_inputs = {value_node,  input,  gamma1, beta1,    weight_qkv, bias_attn_qkv, input_mask, weight_attn_o,
                         bias_attn_o, gamma2, beta2,  weight_m, bias_m,     weight_p,      bias_p};
    } else {
      new_node_inputs = {value_node,    input,       weight_qkv, bias_attn_qkv, input_mask,
                         weight_attn_o, bias_attn_o, gamma1,     beta1,         weight_m,
                         bias_m,        weight_p,    bias_p,     gamma2,        beta2};
    }
  }
  auto new_node = func_graph->NewCNode(new_node_inputs);
  MS_CHECK_TRUE_RET(new_node != nullptr, nullptr);
  auto old_node = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(old_node->abstract() != nullptr, nullptr);
  new_node->set_abstract(old_node->abstract()->Clone());
  new_node->set_fullname_with_scope(node->fullname_with_scope() + "/encoder_layer");
  return new_node;
}
}  // namespace mindspore::opt
