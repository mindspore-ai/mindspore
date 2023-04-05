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
  expert_ids_ = std::make_shared<Var>("expert_ids");
  MS_CHECK_TRUE_RET(expert_ids_ != nullptr, false);
  expert_capacity_ = std::make_shared<Var>("expert_capacity_");
  MS_CHECK_TRUE_RET(expert_capacity_ != nullptr, false);
  begin_expert_ids_ = std::make_shared<Var>("weight_m");
  MS_CHECK_TRUE_RET(begin_expert_ids_ != nullptr, false);
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
  weight_attn_qkv_ = std::make_shared<Var>("weight_attn_qkv");
  MS_CHECK_TRUE_RET(weight_attn_qkv_ != nullptr, false);
  weight_attn_q_ = std::make_shared<Var>("weight_attn_q_");
  MS_CHECK_TRUE_RET(weight_attn_q_ != nullptr, false);
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
  MS_CHECK_TRUE_RET(position_bias_ != nullptr, false);
  is_act_ = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimActivation), "activation");
  MS_CHECK_TRUE_RET(is_act_ != nullptr, {});
  eps1_ = std::make_shared<Var>("eps1_");
  MS_CHECK_TRUE_RET(eps1_ != nullptr, false);
  eps2_ = std::make_shared<Var>("eps2_");
  MS_CHECK_TRUE_RET(eps2_ != nullptr, false);
  eps3_ = std::make_shared<Var>("eps3_");
  MS_CHECK_TRUE_RET(eps3_ != nullptr, false);
  batch_valid_length_ = std::make_shared<Var>("batch_valid_length");
  MS_CHECK_TRUE_RET(batch_valid_length_ != nullptr, false);
  k_past_ = std::make_shared<Var>("k_past");
  MS_CHECK_TRUE_RET(k_past_ != nullptr, false);
  v_past_ = std::make_shared<Var>("k_past");
  MS_CHECK_TRUE_RET(v_past_ != nullptr, false);
  init_reset_ = std::make_shared<Var>("init_reset");
  MS_CHECK_TRUE_RET(init_reset_ != nullptr, false);
  input_q_ = std::make_shared<Var>("input_q");
  MS_CHECK_TRUE_RET(input_q_ != nullptr, false);
  embedding_table_ = std::make_shared<Var>("embedding_table");
  MS_CHECK_TRUE_RET(embedding_table_ != nullptr, false);
  is_layernorm3_ = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimLayerNormFusion), "layer_norm2");
  MS_CHECK_TRUE_RET(is_layernorm3_ != nullptr, false);
  return true;
}

VectorRef EncoderLayerFusion::getTuple(bool post_layernorm, bool layernorm_fusion = false,
                                       bool is_position_bias = false) const {
  auto is_reshape1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "reshape-encoder");
  MS_CHECK_TRUE_RET(is_reshape1 != nullptr, {});
  auto var1 = std::make_shared<Var>("var1-reshape");
  MS_CHECK_TRUE_RET(var1 != nullptr, {});
  auto reshape1 = VectorRef({is_reshape1, input_, var1});
  if (post_layernorm && !is_position_bias) {
    return reshape1;
  }
  if (!layernorm_fusion) {
    return DefineLayerNorm(is_position_bias, reshape1, gamma1_, beta1_, eps1_, false);
  }
  auto layer_norm = VectorRef({is_layernorm1_, reshape1, gamma1_, beta1_});
  auto is_tuple = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTupleGetItem), "tuple_get_itme");
  auto var_tuple = std::make_shared<Var>("var_tuple");
  auto tuple = VectorRef({is_tuple, layer_norm, var_tuple});
  return tuple;
}

VectorRef EncoderLayerFusion::DefineLayerNorm(bool is_position_bias, BaseRef input, VarPtr gamma, VarPtr beta,
                                              VarPtr eps, bool sigma = false) const {
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
  auto is_add = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAddFusion), "is-add");
  MS_CHECK_TRUE_RET(is_add != nullptr, {});
  auto add = VectorRef({is_add, reduce2, eps});
  auto is_sqr2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimSqrt), "sqr2");
  MS_CHECK_TRUE_RET(is_sqr2 != nullptr, {});
  auto sqr2 = VectorRef({is_sqr2, add});
  auto is_div = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimRealDiv), "real-div");
  MS_CHECK_TRUE_RET(is_div != nullptr, {});
  auto is_mul = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMulFusion), "mul");
  MS_CHECK_TRUE_RET(is_mul != nullptr, {});
  if (is_position_bias) {
    auto real_div = VectorRef({is_div, input, sqr2});
    auto mul = VectorRef({is_mul, real_div, gamma});
    return mul;
  } else {
    auto real_div = VectorRef({is_div, sub, sqr2});
    if (sigma) {
      auto mul = VectorRef({is_mul, real_div, gamma});
      auto is_add2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAddFusion), "is-add");
      MS_CHECK_TRUE_RET(is_add2 != nullptr, {});
      auto add2 = VectorRef({is_add2, mul, beta});
      return add2;
    }
    auto is_scale = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimScaleFusion), "scale");
    MS_CHECK_TRUE_RET(is_scale != nullptr, {});
    auto scale = VectorRef({is_scale, real_div, gamma, beta});
    return scale;
  }
}

VectorRef EncoderLayerFusion::DefinePatternInitReset(VectorRef input, bool value_reset, bool key_reset) const {
  auto is_cast = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimCast), "cast_init");
  auto var_cast = std::make_shared<Var>("var_cast");
  auto cast = VectorRef({is_cast, init_reset_, var_cast});
  auto is_mul_k = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMulFusion), "mul_k");
  auto mul_k = VectorRef({is_mul_k, k_past_, cast});
  auto is_assign_k = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAssign), "assign_k");
  auto var_assign_k = std::make_shared<Var>("var_assign");
  auto assign_k = VectorRef({is_assign_k, k_past_, mul_k, var_assign_k});
  if (key_reset) return assign_k;
  auto is_depend_k = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimDepend), "depend_k");
  auto depend_k = VectorRef({is_depend_k, input, assign_k});
  auto is_mul_v = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMulFusion), "mul_v");
  auto mul_v = VectorRef({is_mul_v, v_past_, cast});
  auto is_assign_v = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAssign), "assign_v");
  auto var_assign_v = std::make_shared<Var>("var_assign");
  auto assign_v = VectorRef({is_assign_v, v_past_, mul_v, var_assign_v});
  if (value_reset) return assign_v;
  auto is_depend_kv = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimDepend), "depend_kv");
  auto depend_kv = VectorRef({is_depend_kv, depend_k, assign_v});
  return depend_kv;
}

BaseRef EncoderLayerFusion::DefineBatchValidLength(const BaseRef &input) const {
  auto is_reshape = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "is_reshpae");
  MS_CHECK_TRUE_RET(is_reshape != nullptr, {});
  auto var = std::make_shared<Var>("var");
  MS_CHECK_TRUE_RET(var != nullptr, {});
  auto reshape = VectorRef({is_reshape, batch_valid_length_, var});
  auto var2 = std::make_shared<Var>("var2");
  MS_CHECK_TRUE_RET(var2 != nullptr, {});
  auto is_less = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimLess), "is_less");
  MS_CHECK_TRUE_RET(is_less != nullptr, {});
  auto less = VectorRef({is_less, var2, reshape});
  auto var3 = std::make_shared<Var>("var3");
  MS_CHECK_TRUE_RET(var3 != nullptr, {});
  auto is_cast = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimCast), "is_cast");
  MS_CHECK_TRUE_RET(is_cast != nullptr, {});
  auto cast = VectorRef({is_cast, less, var3});
  auto var4 = std::make_shared<Var>("var4");
  MS_CHECK_TRUE_RET(var4 != nullptr, {});
  auto is_expand_dims = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimExpandDims), "is_expand_dims");
  MS_CHECK_TRUE_RET(is_expand_dims != nullptr, {});
  auto expand_dims = VectorRef({is_expand_dims, cast, var4});
  auto is_mul = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMulFusion), "is_mul");
  MS_CHECK_TRUE_RET(is_mul != nullptr, {});
  auto mul = VectorRef({is_mul, input, expand_dims});
  return mul;
}

VectorRef EncoderLayerFusion::DefinePatternMoETopKRouter(VectorRef input) const {
  auto var_onehot1 = std::make_shared<Var>("var_onehot1");
  auto var_onehot2 = std::make_shared<Var>("var_onehot2");
  auto var_onehot3 = std::make_shared<Var>("var_onehot3");
  auto var_onehot4 = std::make_shared<Var>("var_onehot4");
  auto is_onehot = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimOneHot), "is_onehot");
  MS_CHECK_TRUE_RET(is_onehot != nullptr, {});
  auto onehot = VectorRef({is_onehot, input, var_onehot1, var_onehot2, var_onehot3});
  auto is_transpose3 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTranspose), "is_transpose1");
  MS_CHECK_TRUE_RET(is_transpose3 != nullptr, {});
  auto var_transpose3 = std::make_shared<Var>("var_transpose3");
  MS_CHECK_TRUE_RET(var_transpose3 != nullptr, {});
  auto transpose3 = VectorRef({is_transpose3, onehot, var_transpose3});
  auto is_reshape1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "is_reshape1");
  MS_CHECK_TRUE_RET(is_reshape1 != nullptr, {});
  auto var_reshape1 = std::make_shared<Var>("var_reshape1");
  MS_CHECK_TRUE_RET(var_reshape1 != nullptr, {});
  auto reshape1 = VectorRef({is_reshape1, transpose3, var_reshape1});
  auto is_matmul = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMulFusion), "router-matmul");
  MS_CHECK_TRUE_RET(is_matmul != nullptr, {});
  auto var_matmul1 = std::make_shared<Var>("var_matmul1");
  auto matmul1 = VectorRef({is_matmul, reshape1, var_matmul1});
  auto is_reshape2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "is_reshape1");
  MS_CHECK_TRUE_RET(is_reshape2 != nullptr, {});
  auto var_reshape2 = std::make_shared<Var>("var_reshape2");
  MS_CHECK_TRUE_RET(var_reshape2 != nullptr, {});
  auto reshape2 = VectorRef({is_reshape2, matmul1, var_reshape2});
  auto is_transpose2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTranspose), "is_transpose2");
  MS_CHECK_TRUE_RET(is_transpose2 != nullptr, {});
  auto var_transpose2 = std::make_shared<Var>("var_transpose2");
  MS_CHECK_TRUE_RET(var_transpose2 != nullptr, {});
  auto transpose2 = VectorRef({is_transpose2, reshape2, var_transpose2});
  auto is_mul1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMulFusion), "is_mul1");
  MS_CHECK_TRUE_RET(is_mul1 != nullptr, {});
  auto mul_fusion1 = VectorRef({is_mul1, transpose2, onehot});
  auto is_less = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimLess), "is_less");
  MS_CHECK_TRUE_RET(is_less != nullptr, {});
  auto less = VectorRef({is_less, mul_fusion1, expert_capacity_});
  auto is_cast3 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimCast), "is_cast3");
  MS_CHECK_TRUE_RET(is_cast3 != nullptr, {});
  auto var_cast3 = std::make_shared<Var>("var_cast3");
  auto cast3 = VectorRef({is_cast3, less, var_cast3});
  auto is_mul4 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMulFusion), "is_mul4");
  MS_CHECK_TRUE_RET(is_mul4 != nullptr, {});
  auto mul_fusion4 = VectorRef({is_mul4, cast3, onehot});
  auto is_reduce5 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReduceFusion), "is_reduce5");
  MS_CHECK_TRUE_RET(is_reduce5 != nullptr, {});
  auto var_reduce5 = std::make_shared<Var>("var_reduce5");
  auto reduce5 = VectorRef({is_reduce5, mul_fusion4, var_reduce5});
  auto is_expand_dims1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimExpandDims), "is_expand_dims1");
  MS_CHECK_TRUE_RET(is_expand_dims1 != nullptr, {});
  auto var_expand_dims = std::make_shared<Var>("var_expand_dims1");
  auto expand_dims1 = VectorRef({is_expand_dims1, reduce5, var_expand_dims});
  auto is_mul7 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMulFusion), "is_mul7");
  MS_CHECK_TRUE_RET(is_mul7 != nullptr, {});
  auto mul_fusion7 = VectorRef({is_mul7, expand_dims1, onehot});
  auto is_onehot2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimOneHot), "is_onehot2");
  MS_CHECK_TRUE_RET(is_onehot2 != nullptr, {});
  auto is_cast2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimCast), "is_cast2");
  MS_CHECK_TRUE_RET(is_cast2 != nullptr, {});
  auto var_cast2 = std::make_shared<Var>("var_cast2");
  auto cast2 = VectorRef({is_cast2, mul_fusion1, var_cast2});
  auto onehot2 = VectorRef({is_onehot2, cast2, var_onehot4, var_onehot2, var_onehot3});
  auto is_expand_dims2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimExpandDims), "is_expand_dims2");
  MS_CHECK_TRUE_RET(is_expand_dims2 != nullptr, {});
  auto expand_dims2 = VectorRef({is_expand_dims2, mul_fusion7, var_expand_dims});
  auto is_mul8 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMulFusion), "is_mul8");
  MS_CHECK_TRUE_RET(is_mul8 != nullptr, {});
  auto mul_fusion8 = VectorRef({is_mul8, expand_dims2, onehot2});
  return mul_fusion8;
}

VectorRef EncoderLayerFusion::DefinePatternMoERouter() const {
  auto is_stride_slice = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimStridedSlice), "is_stride_slice");
  MS_CHECK_TRUE_RET(is_stride_slice != nullptr, {});
  auto var_stride_slice1 = std::make_shared<Var>("var_stride_slice1");
  MS_CHECK_TRUE_RET(var_stride_slice1 != nullptr, {});
  auto var_stride_slice2 = std::make_shared<Var>("var_stride_slice2");
  MS_CHECK_TRUE_RET(var_stride_slice2 != nullptr, {});
  auto var_stride_slice3 = std::make_shared<Var>("var_stride_slice3");
  MS_CHECK_TRUE_RET(var_stride_slice3 != nullptr, {});
  auto strid_slice = VectorRef({is_stride_slice, expert_ids_, begin_expert_ids_, var_stride_slice2, var_stride_slice3});
  auto is_reshape_router = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "router-reshpae");
  MS_CHECK_TRUE_RET(is_reshape_router != nullptr, {});
  auto var1 = std::make_shared<Var>("var1");
  MS_CHECK_TRUE_RET(var1 != nullptr, {});
  auto router_reshape = VectorRef({is_reshape_router, strid_slice, var1});
  return DefinePatternMoETopKRouter(router_reshape);
}

VectorRef EncoderLayerFusion::DefinePatternMoEFfn(VectorRef input_deppend, VectorRef input_reshape) const {
  auto is_matmul1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMulFusion), "is_matmul1");
  auto matmul1 = VectorRef({is_matmul1, input_reshape, input_deppend});
  auto is_add1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAddFusion), "is_add1");
  MS_CHECK_TRUE_RET(is_add1 != nullptr, {});
  auto add = VectorRef({is_add1, matmul1, bias_m_});
  auto is_act = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimFastGeLU), "is_FastGelu");
  auto act = VectorRef({is_act, add});
  auto is_deppend = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimDepend), "depend");
  auto deppend = VectorRef({is_deppend, weight_p_, act});
  auto is_matmul2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMulFusion), "is_matmul2");
  MS_CHECK_TRUE_RET(is_matmul2 != nullptr, {});
  auto matmul2 = VectorRef({is_matmul2, act, deppend});
  auto is_all_reduce = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAllReduce), "is_all_reduce");
  auto all_reduce = VectorRef({is_all_reduce, matmul2});
  auto is_add2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAddFusion), "is_add2");
  MS_CHECK_TRUE_RET(is_add2 != nullptr, {});
  auto add2 = VectorRef({is_add2, all_reduce, bias_p_});
  return add2;
}

VectorRef EncoderLayerFusion::DefinePatternMoE(VectorRef input_layernorm) const {
  auto router_output = DefinePatternMoERouter();
  auto is_reshape10 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "is_reshape1");
  MS_CHECK_TRUE_RET(is_reshape10 != nullptr, {});
  auto var_reshape10 = std::make_shared<Var>("var_reshape10");
  MS_CHECK_TRUE_RET(var_reshape10 != nullptr, {});
  auto reshape10 = VectorRef({is_reshape10, router_output, var_reshape10});
  auto is_matmul1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMulFusion), "matmul168");
  MS_CHECK_TRUE_RET(is_matmul1 != nullptr, {});
  auto matmul1 = VectorRef({is_matmul1, input_layernorm, reshape10});
  auto is_reshape3 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "is_reshape3");
  MS_CHECK_TRUE_RET(is_reshape3 != nullptr, {});
  auto var_reshape3 = std::make_shared<Var>("var_reshape3");
  MS_CHECK_TRUE_RET(var_reshape3 != nullptr, {});
  auto reshape3 = VectorRef({is_reshape3, matmul1, var_reshape3});
  auto is_transpose1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTranspose), "is_transpose1");
  MS_CHECK_TRUE_RET(is_transpose1 != nullptr, {});
  auto var_transpose1 = std::make_shared<Var>("var_transpose1");
  MS_CHECK_TRUE_RET(var_reshape3 != nullptr, {});
  auto transpose1 = VectorRef({is_transpose1, reshape3, var_transpose1});
  auto is_reshape4 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "is_reshape4");
  MS_CHECK_TRUE_RET(is_reshape4 != nullptr, {});
  auto var_reshape4 = std::make_shared<Var>("var_reshape4");
  MS_CHECK_TRUE_RET(var_reshape4 != nullptr, {});
  auto reshape4 = VectorRef({is_reshape4, transpose1, var_reshape4});
  auto is_reshape5 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "is_reshape5");
  MS_CHECK_TRUE_RET(is_reshape5 != nullptr, {});
  auto var_reshape5 = std::make_shared<Var>("var_reshape5");
  MS_CHECK_TRUE_RET(var_reshape5 != nullptr, {});
  auto reshape5 = VectorRef({is_reshape5, reshape4, var_reshape5});
  auto is_deppend = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimDepend), "depend");
  auto deppend = VectorRef({is_deppend, weight_m_, reshape5});
  auto ffn_output = DefinePatternMoEFfn(deppend, reshape5);
  auto is_reshape6 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "is_reshape6");
  MS_CHECK_TRUE_RET(is_reshape6 != nullptr, {});
  auto var_reshape6 = std::make_shared<Var>("var_reshape6");
  MS_CHECK_TRUE_RET(var_reshape6 != nullptr, {});
  auto reshape6 = VectorRef({is_reshape6, ffn_output, var_reshape6});
  auto is_transpose2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTranspose), "is_transpose2");
  MS_CHECK_TRUE_RET(is_transpose2 != nullptr, {});
  auto var_transpose2 = std::make_shared<Var>("var_transpose2");
  MS_CHECK_TRUE_RET(var_transpose2 != nullptr, {});
  auto transpose2 = VectorRef({is_transpose2, reshape6, var_transpose2});
  auto is_reshape7 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "is_reshape7");
  MS_CHECK_TRUE_RET(is_reshape7 != nullptr, {});
  auto var_reshape7 = std::make_shared<Var>("var_reshape7");
  MS_CHECK_TRUE_RET(var_reshape7 != nullptr, {});
  auto reshape7 = VectorRef({is_reshape7, transpose2, var_reshape7});
  auto is_transpose3 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTranspose), "is_transpose3");
  MS_CHECK_TRUE_RET(is_transpose3 != nullptr, {});
  auto var_transpose3 = std::make_shared<Var>("var_transpose3");
  MS_CHECK_TRUE_RET(var_transpose3 != nullptr, {});
  auto transpose3 = VectorRef({is_transpose3, reshape7, var_transpose3});
  auto is_reshape8 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "is_reshape8");
  MS_CHECK_TRUE_RET(is_reshape8 != nullptr, {});
  auto var_reshape8 = std::make_shared<Var>("var_reshape8");
  MS_CHECK_TRUE_RET(var_reshape8 != nullptr, {});
  auto reshape8 = VectorRef({is_reshape8, transpose3, var_reshape8});
  auto is_matmul2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMulFusion), "matmul2");
  MS_CHECK_TRUE_RET(is_matmul2 != nullptr, {});
  auto matmul2 = VectorRef({is_matmul2, reshape8, reshape10});
  auto is_transpose4 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTranspose), "is_transpose4");
  MS_CHECK_TRUE_RET(is_transpose4 != nullptr, {});
  auto var_transpose4 = std::make_shared<Var>("var_transpose4");
  MS_CHECK_TRUE_RET(var_transpose4 != nullptr, {});
  auto transpose4 = VectorRef({is_transpose4, matmul2, var_transpose4});
  return transpose4;
}

VectorRef EncoderLayerFusion::DefineDependKV(VectorRef input, VectorRef deppend_v_input,
                                             bool is_distributed = true) const {
  auto is_tuple_k = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTupleGetItem), "tuple_get_itme");
  auto var_tuple_k = std::make_shared<Var>("var_tuple");
  auto tuple_k = VectorRef({is_tuple_k, input, var_tuple_k});
  if (is_distributed) {
    auto is_reshape_k1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "reshape-encoder_k1");
    MS_CHECK_TRUE_RET(is_reshape_k1 != nullptr, {});
    auto var_k1 = std::make_shared<Var>("var_k1");
    MS_CHECK_TRUE_RET(var_k1 != nullptr, {});
    auto reshape_k1 = VectorRef({is_reshape_k1, tuple_k, var_k1});
    auto is_all_gather_k = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAllGather), "is_all_gather_k");
    MS_CHECK_TRUE_RET(is_all_gather_k != nullptr, {});
    auto all_gather_k = VectorRef({is_all_gather_k, reshape_k1});
    auto is_reshape_k2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "reshape-encoder_k2");
    MS_CHECK_TRUE_RET(is_reshape_k2 != nullptr, {});
    auto var_k2 = std::make_shared<Var>("var_k2");
    MS_CHECK_TRUE_RET(var_k2 != nullptr, {});
    auto reshape_k2 = VectorRef({is_reshape_k2, all_gather_k, var_k2});
    tuple_k = reshape_k2;
  }
  auto is_assign_k = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAssign), "assign_k_fuse");
  auto var_assign_k = std::make_shared<Var>("var_assign_k");
  auto assign_k = VectorRef({is_assign_k, k_past_, DefineBatchValidLength(tuple_k), var_assign_k});
  auto is_depend_k = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimDepend), "is_depend_k");
  auto update_k = VectorRef({is_depend_k, assign_k, DefinePatternInitReset(assign_k, false, true)});
  auto is_tuple_v = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTupleGetItem), "tuple_get_itme");
  auto var_tuple_v = std::make_shared<Var>("var_tuple");
  auto tuple_v = VectorRef({is_tuple_v, input, var_tuple_v});
  if (is_distributed) {
    auto is_reshape_v1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "reshape-encoder_v1");
    MS_CHECK_TRUE_RET(is_reshape_v1 != nullptr, {});
    auto var_v1 = std::make_shared<Var>("var_v1");
    MS_CHECK_TRUE_RET(var_v1 != nullptr, {});
    auto reshape_v1 = VectorRef({is_reshape_v1, tuple_v, var_v1});
    auto is_all_gather_v = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAllGather), "is_all_gather_v");
    MS_CHECK_TRUE_RET(is_all_gather_v != nullptr, {});
    auto all_gather_v = VectorRef({is_all_gather_v, reshape_v1});
    auto is_reshape_v2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "reshape-encoder_v2");
    MS_CHECK_TRUE_RET(is_reshape_v2 != nullptr, {});
    auto var_v2 = std::make_shared<Var>("var_v2");
    MS_CHECK_TRUE_RET(var_v2 != nullptr, {});
    auto reshape_v2 = VectorRef({is_reshape_v2, all_gather_v, var_v2});
    tuple_v = reshape_v2;
  }
  auto is_assign_v = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAssign), "assign_k_fuse");
  auto var_assign_v = std::make_shared<Var>("var_assign_v");
  auto assign_v = VectorRef({is_assign_v, v_past_, DefineBatchValidLength(tuple_v), var_assign_v});
  auto is_depend_v = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimDepend), "depend_kv");
  auto update_v = VectorRef({is_depend_v, assign_v, DefinePatternInitReset(assign_v, true)});
  auto is_depend_v_mul = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimDepend), "depend_kv");
  auto depend_v_mul = VectorRef({is_depend_v_mul, deppend_v_input, update_v});
  auto is_depend_kv_mul = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimDepend), "depend_kv");
  return VectorRef({is_depend_kv_mul, depend_v_mul, update_k});
}

VectorRef EncoderLayerFusion::DefinePatternSigmaFfn(BaseRef input) const {
  auto is_reshape2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "reshape-encoder2");
  MS_CHECK_TRUE_RET(is_reshape2 != nullptr, {});
  auto var2 = std::make_shared<Var>("var2");
  MS_CHECK_TRUE_RET(var2 != nullptr, {});
  auto reshape2 = VectorRef({is_reshape2, input, var2});
  auto is_matmul1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMulFusion), "is_matmul1");
  auto matmul1 = VectorRef({is_matmul1, reshape2, weight_m_, bias_m_});
  auto is_reshape3 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "reshape-encoder3");
  MS_CHECK_TRUE_RET(is_reshape3 != nullptr, {});
  auto var3 = std::make_shared<Var>("var3");
  MS_CHECK_TRUE_RET(var3 != nullptr, {});
  auto reshape3 = VectorRef({is_reshape3, matmul1, var3});
  auto is_act = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimFastGeLU), "is_FastGelu");
  MS_CHECK_TRUE_RET(is_act != nullptr, {});
  auto act = VectorRef({is_act, reshape3});
  auto is_reshape4 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "reshape-encoder4");
  MS_CHECK_TRUE_RET(is_reshape4 != nullptr, {});
  auto var4 = std::make_shared<Var>("var4");
  MS_CHECK_TRUE_RET(var4 != nullptr, {});
  auto reshape4 = VectorRef({is_reshape4, act, var4});
  auto is_matmul2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMulFusion), "is_matmul2");
  MS_CHECK_TRUE_RET(is_matmul2 != nullptr, {});
  auto matmul2 = VectorRef({is_matmul2, reshape4, weight_p_});
  auto is_all_reduce = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAllReduce), "is_all_reduce");
  auto all_reduce = VectorRef({is_all_reduce, matmul2});
  auto is_add2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAddFusion), "is_add2");
  auto add2 = VectorRef({is_add2, all_reduce, bias_p_});
  auto is_reshape5 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "reshape-encoder5");
  MS_CHECK_TRUE_RET(is_reshape5 != nullptr, {});
  auto var5 = std::make_shared<Var>("var5");
  MS_CHECK_TRUE_RET(var5 != nullptr, {});
  auto reshape5 = VectorRef({is_reshape5, add2, var5});
  return reshape5;
}

VectorRef EncoderLayerFusion::DefinePatternEncoderSigma(bool moe = false, bool use_past = true,
                                                        bool distributed = false, bool is_layer_norm = false,
                                                        bool query_layer = false) const {
  auto is_reshape = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "reshape-encoder");
  MS_CHECK_TRUE_RET(is_reshape != nullptr, {});
  auto var = std::make_shared<Var>("var");
  MS_CHECK_TRUE_RET(var != nullptr, {});
  auto reshape =
    VectorRef({is_reshape, DefinePatternInitReset(DefineLayerNorm(false, input_, gamma1_, beta1_, eps1_, true)), var});
  auto attention = (query_layer) ? VectorRef({is_attention_, input_q_, reshape, reshape, weight_attn_q_,
                                              weight_attn_qkv_, weight_attn_o_, bias_attn_qkv_, bias_attn_o_, mask_})
                                 : VectorRef({is_attention_, reshape, reshape, reshape, weight_attn_qkv_,
                                              weight_attn_o_, bias_attn_qkv_, bias_attn_o_, mask_});
  auto is_tuple = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTupleGetItem), "tuple_get_itme");
  auto var_tuple = std::make_shared<Var>("var_tuple");
  auto tuple = VectorRef({is_tuple, attention, var_tuple});
  auto is_add = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAddFusion), "is_add");
  auto add = VectorRef({is_add, input_, tuple});
  auto layer_norm2 = DefineLayerNorm(false, add, gamma2_, beta2_, eps2_, true);
  auto ffn_output = (moe) ? DefinePatternMoE(layer_norm2) : DefinePatternSigmaFfn(layer_norm2);
  auto depend_kv_mul = DefineDependKV(attention, ffn_output);
  auto is_add3 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAddFusion), "is_add3");
  auto add3 = VectorRef({is_add3, add, depend_kv_mul});
  if (is_layer_norm) return DefineLayerNorm(false, add3, gamma3_, beta3_, eps3_, true);
  if (query_layer) {
    if (moe) {
      auto is_reshape7 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "reshape-encoder7");
      MS_CHECK_TRUE_RET(is_reshape7 != nullptr, {});
      auto var7 = std::make_shared<Var>("var7");
      MS_CHECK_TRUE_RET(var7 != nullptr, {});
      auto reshape7 = VectorRef({is_reshape7, add3, var7});
      add3 = reshape7;
    }
    auto is_matmul3 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMulFusion), "is_matmul3");
    MS_CHECK_TRUE_RET(is_matmul3 != nullptr, {});
    auto matmul3 = VectorRef({is_matmul3, add3, embedding_table_});
    return matmul3;
  }
  return add3;
}

VectorRef EncoderLayerFusion::DefinePatternEncoderAlpha(bool moe = false, bool distributed = false,
                                                        bool is_layer_norm = false, bool query_layer = false,
                                                        bool use_past = false) const {
  VectorRef add2;
  auto is_reshape1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "reshape-encoder");
  MS_CHECK_TRUE_RET(is_reshape1 != nullptr, {});
  auto var1 = std::make_shared<Var>("var1-reshape");
  MS_CHECK_TRUE_RET(var1 != nullptr, {});
  auto reshape1 = VectorRef({is_reshape1, input_, var1});
  VectorRef layer_norm = (query_layer) ? VectorRef({is_layernorm1_, input_, gamma1_, beta1_})
                                       : VectorRef({is_layernorm1_, reshape1, gamma1_, beta1_});
  auto is_tuple = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTupleGetItem), "tuple_get_itme");
  auto var_tuple = std::make_shared<Var>("var_tuple");
  auto tuple = VectorRef({is_tuple, layer_norm, var_tuple});
  auto is_deppend = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimDepend), "depend");
  auto deppend = VectorRef({is_deppend, tuple, k_past_});
  auto is_deppend2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimDepend), "depend");
  auto deppend2 = VectorRef({is_deppend2, deppend, v_past_});
  auto attention = (query_layer) ? VectorRef({is_attention_, input_q_, DefinePatternInitReset(tuple, false, false),
                                              DefinePatternInitReset(tuple, false, false), weight_attn_q_,
                                              weight_attn_qkv_, weight_attn_o_, bias_attn_qkv_, bias_attn_o_, mask_})
                                 : VectorRef({is_attention_, deppend2, deppend2, deppend2, weight_attn_qkv_,
                                              weight_attn_o_, bias_attn_qkv_, bias_attn_o_, mask_});
  auto is_tuple1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTupleGetItem), "tuple_get_itme");
  auto var_tuple1 = std::make_shared<Var>("var_tuple1");
  auto tuple1 = VectorRef({is_tuple1, attention, var_tuple1});
  auto is_add = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAddFusion), "is_add");
  auto add = (query_layer) ? VectorRef({is_add, input_, tuple1}) : VectorRef({is_add, reshape1, tuple1});
  auto layer_norm2 = VectorRef({is_layernorm2_, add, gamma2_, beta2_});
  auto is_tuple2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTupleGetItem), "tuple_get_item2");
  auto var_tuple2 = std::make_shared<Var>("var_tuple2");
  auto tuple2 = VectorRef({is_tuple2, layer_norm2, var_tuple2});
  auto is_matmul1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMulFusion), "is_matmul1");
  auto matmul1 = VectorRef({is_matmul1, tuple2, weight_m_, bias_m_});
  auto is_act = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimFastGeLU), "is_FastGelu");
  MS_CHECK_TRUE_RET(is_act != nullptr, {});
  auto act = VectorRef({is_act, matmul1});
  auto is_matmul2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMulFusion), "is_matmul2");
  auto matmul2 =
    (distributed) ? VectorRef({is_matmul2, act, weight_p_}) : VectorRef({is_matmul2, act, weight_p_, bias_p_});
  if (distributed) {
    auto is_all_reduce = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAllReduce), "is_all_reduce");
    auto all_reduce = VectorRef({is_all_reduce, matmul2});
    auto is_add2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAddFusion), "is_add2");
    add2 = VectorRef({is_add2, all_reduce, bias_p_});
  }
  if (use_past && !query_layer) {
    auto is_deppend3 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimDepend), "depend3");
    auto deppend3 = VectorRef({is_deppend3, v_past_, v_past_});
    auto is_deppend4 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimDepend), "depend");
    auto deppend4 =
      (distributed) ? VectorRef({is_deppend4, add2, deppend3}) : VectorRef({is_deppend4, matmul2, deppend3});
    auto is_deppend5 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimDepend), "depend");
    auto deppend5 = VectorRef({is_deppend5, k_past_, k_past_});
    auto is_deppend6 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimDepend), "depend");
    matmul2 = VectorRef({is_deppend6, deppend4, deppend5});
  }
  auto is_reshape2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "reshape-encoder2");
  MS_CHECK_TRUE_RET(is_reshape2 != nullptr, {});
  auto var2 = std::make_shared<Var>("var2");
  MS_CHECK_TRUE_RET(var2 != nullptr, {});
  auto reshape2 = VectorRef({is_reshape2, add, var2});
  auto is_reshape3 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "reshape-encoder3");
  MS_CHECK_TRUE_RET(is_reshape3 != nullptr, {});
  auto var3 = std::make_shared<Var>("var3");
  MS_CHECK_TRUE_RET(var3 != nullptr, {});
  auto reshape3 = VectorRef({is_reshape3, matmul2, var3});
  auto depend_kv_mul =
    (distributed) ? DefineDependKV(attention, add2, true) : DefineDependKV(attention, matmul2, false);
  auto is_add3 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAddFusion), "is_add3");
  auto add3 = (query_layer) ? VectorRef({is_add3, add, depend_kv_mul}) : VectorRef({is_add3, reshape2, reshape3});
  if (is_layer_norm) {
    auto is_reshape_norm = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "reshape-encoder3");
    MS_CHECK_TRUE_RET(is_reshape_norm != nullptr, {});
    auto var_norm = std::make_shared<Var>("var3");
    MS_CHECK_TRUE_RET(var_norm != nullptr, {});
    auto reshape_norm = VectorRef({is_reshape_norm, add3, var_norm});
    auto layer_norm3 = VectorRef({is_layernorm3_, reshape_norm, gamma3_, beta3_});
    return layer_norm3;
  }
  if (query_layer) {
    auto is_matmul3 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMulFusion), "is_matmul3");
    MS_CHECK_TRUE_RET(is_matmul3 != nullptr, {});
    auto matmul3 = VectorRef({is_matmul3, add3, embedding_table_});
    return matmul3;
  }
  return add3;
}

VectorRef EncoderLayerFusion::DefinePatternEncoderLayer(bool post_layernorm = true, bool layernorm_fusion = false,
                                                        bool is_position_bias = false, bool mask = true,
                                                        bool is_layer_norm = false) const {
  VectorRef tuple, tuple2, tuple3, reshape2, matmul1, inputs, layer_norm2;
  auto is_reshape1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "reshape-encoder");
  MS_CHECK_TRUE_RET(is_reshape1 != nullptr, {});
  auto var1 = std::make_shared<Var>("var1");
  MS_CHECK_TRUE_RET(var1 != nullptr, {});
  auto reshape1 = VectorRef({is_reshape1, input_, var1});
  if (!is_position_bias) {
    inputs = VectorRef({is_attention_, getTuple(post_layernorm, layernorm_fusion, is_position_bias),
                        getTuple(post_layernorm, layernorm_fusion, is_position_bias),
                        getTuple(post_layernorm, layernorm_fusion, is_position_bias), weight_attn_qkv_, weight_attn_o_,
                        bias_attn_qkv_, bias_attn_o_});
  } else {
    inputs = VectorRef({is_attention_, getTuple(post_layernorm, layernorm_fusion, is_position_bias),
                        getTuple(post_layernorm, layernorm_fusion, is_position_bias),
                        getTuple(post_layernorm, layernorm_fusion, is_position_bias), weight_attn_qkv_, weight_attn_o_,
                        position_bias_});
  }
  if (mask) inputs.push_back(mask_);
  auto attention = VectorRef(inputs);
  if (!is_position_bias) {
    auto is_tuple = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTupleGetItem), "tuple_get_itme");
    auto var_tuple = std::make_shared<Var>("var_tuple");
    tuple = VectorRef({is_tuple, attention, var_tuple});
  } else {
    tuple = attention;
  }
  auto is_add = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAddFusion), "is_add");
  auto add = (is_position_bias && post_layernorm)
               ? VectorRef({is_add, getTuple(post_layernorm, layernorm_fusion, is_position_bias), tuple})
               : VectorRef({is_add, reshape1, tuple});
  auto is_reshape2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "reshape-encoder2");
  MS_CHECK_TRUE_RET(is_reshape2 != nullptr, {});
  auto var2 = std::make_shared<Var>("var2");
  MS_CHECK_TRUE_RET(var2 != nullptr, {});
  if (layernorm_fusion) {
    layer_norm2 = VectorRef({is_layernorm2_, add, gamma2_, beta2_});
    auto is_tuple2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTupleGetItem), "tuple_get_item2");
    auto var_tuple2 = std::make_shared<Var>("var_tuple2");
    tuple2 = VectorRef({is_tuple2, layer_norm2, var_tuple2});
  } else {
    tuple2 = DefineLayerNorm(is_position_bias, add, gamma2_, beta2_, eps2_, false);
  }
  auto is_matmul1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMulFusion), "is_matmul1");
  MS_CHECK_TRUE_RET(is_matmul1 != nullptr, {});
  if (is_position_bias) {
    reshape2 = (post_layernorm) ? VectorRef({is_reshape2, tuple2, var2}) : VectorRef({is_reshape2, add, var2});
    matmul1 = VectorRef({is_matmul1, tuple2, weight_m_});
  } else if (post_layernorm || !layernorm_fusion) {
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
  if (is_layer_norm) return DefineLayerNorm(is_position_bias, add3, gamma3_, beta3_, eps3_);
  if (!post_layernorm || !layernorm_fusion) return add3;
  auto is_reshape4 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "reshape-encoder");
  MS_CHECK_TRUE_RET(is_reshape4 != nullptr, {});
  auto var4 = std::make_shared<Var>("var4");
  MS_CHECK_TRUE_RET(var4 != nullptr, {});
  auto reshape4 = VectorRef({is_reshape4, add3, var4});
  if (layernorm_fusion) {
    auto layer_norm = VectorRef({is_layernorm1_, reshape4, gamma1_, beta1_});
    auto is_tuple3 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTupleGetItem), "tuple_get_item3");
    auto var_tuple3 = std::make_shared<Var>("var_tuple3");
    tuple3 = VectorRef({is_tuple3, layer_norm, var_tuple3});
  } else {
    tuple3 = DefineLayerNorm(is_position_bias, reshape4, gamma1_, beta1_, eps1_);
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
  // pangu sigma
  patterns[kPatternSigmaDistributed] = DefinePatternEncoderSigma(false, true, true, false, false);
  patterns[kPatternSigmaMoeDistributed] = DefinePatternEncoderSigma(true, true, true, false, false);
  patterns[kPatternSigmaMoeWithLastLayerNormDistributed] = DefinePatternEncoderSigma(true, true, true, true, false);
  patterns[kPatternSigmaQueryLayerDistributed] = DefinePatternEncoderSigma(true, true, true, false, true);
  patterns[kPatternSigma] = DefinePatternEncoderSigma(false, true, false, false, false);
  patterns[kPatternSigmaMoe] = DefinePatternEncoderSigma(true, true, false, false, false);
  patterns[kPatternSigmaWithLastLayerNorm] = DefinePatternEncoderSigma(false, true, true, false, false);
  patterns[kPatternSigmaMoeWithLastLayerNorm] = DefinePatternEncoderSigma(true, true, true, true, false);
  patterns[kPatternSigmaQueryLayerMoe] = DefinePatternEncoderSigma(true, true, false, false, true);
  patterns[kPatternSigmaWithLastLayerNormDistributed] = DefinePatternEncoderSigma(false, true, true, true, false);
  // pangu alpha
  patterns[kPatternDistributedAlpha] = DefinePatternEncoderAlpha(false, true, false, false, true);
  patterns[kPatternDistributedAlphaWithLastLayerNorm] = DefinePatternEncoderAlpha(false, true, true, false, true);
  patterns[kPatternQueryLayerUsePastDistributed] = DefinePatternEncoderAlpha(false, true, false, true, true);
  patterns[kPatternQueryLayerUsePast] = DefinePatternEncoderAlpha(false, false, false, true, true);
  patterns[kPatternEncoderLayerPreNormUsePast] = DefinePatternEncoderAlpha(false, false, false, false, true);
  patterns[kPatternEncoderLayerUsePastWithLastNorm] = DefinePatternEncoderAlpha(false, false, true, false, true);
  patterns[kPatternEncoderLayerNormT5Pre] = DefinePatternEncoderLayer(false, false, true, true, true);
  patterns[kPatternEncoderLayerPre] = DefinePatternEncoderLayer(false);
  patterns[kPatternEncoderLayerPost] = DefinePatternEncoderLayer(true);
  patterns[kPatternEncoderLayerPostNorm] = DefinePatternEncoderLayer(true, true);
  patterns[kPatternEncoderLayerPreNorm] = DefinePatternEncoderLayer(false, true);
  patterns[kPatternEncoderLayerT5Pre] = DefinePatternEncoderLayer(false, false, true, true);
  patterns[kPatternEncoderLayerT5Post] = DefinePatternEncoderLayer(true, false, true, true);
  return patterns;
}

bool EncoderLayerFusion::IsUsePast(const std::string &pattern_name) const {
  if (pattern_name == kPatternQueryLayerUsePast || pattern_name == kPatternEncoderLayerPreNormUsePast ||
      pattern_name == kPatternEncoderLayerUsePastWithLastNorm || pattern_name == kPatternSigmaDistributed ||
      pattern_name == kPatternSigmaMoeDistributed || pattern_name == kPatternDistributedAlpha ||
      pattern_name == kPatternQueryLayerUsePastDistributed ||
      pattern_name == kPatternDistributedAlphaWithLastLayerNorm ||
      pattern_name == kPatternSigmaWithLastLayerNormDistributed || pattern_name == kPatternSigmaQueryLayerDistributed ||
      pattern_name == kPatternSigmaMoeWithLastLayerNormDistributed || pattern_name == kPatternSigma ||
      pattern_name == kPatternSigmaMoe || pattern_name == kPatternSigmaMoeWithLastLayerNorm ||
      pattern_name == kPatternSigmaWithLastLayerNorm || pattern_name == kPatternSigmaQueryLayerMoe)
    return true;
  return false;
}

bool EncoderLayerFusion::IsLastLayerNorm(const std::string &pattern_name) const {
  if (pattern_name == kPatternEncoderLayerNormT5Pre || pattern_name == kPatternEncoderLayerUsePastWithLastNorm ||
      pattern_name == kPatternDistributedAlphaWithLastLayerNorm ||
      pattern_name == kPatternSigmaWithLastLayerNormDistributed ||
      pattern_name == kPatternSigmaMoeWithLastLayerNormDistributed ||
      pattern_name == kPatternSigmaMoeWithLastLayerNorm || pattern_name == kPatternSigmaWithLastLayerNorm)
    return true;
  return false;
}

bool EncoderLayerFusion::IsLayerNormFusion(const std::string &pattern_name) const {
  if (pattern_name == kPatternEncoderLayerPostNorm || pattern_name == kPatternEncoderLayerPreNorm ||
      pattern_name == kPatternEncoderLayerPreNormUsePast || pattern_name == kPatternQueryLayerUsePast ||
      pattern_name == kPatternEncoderLayerUsePastWithLastNorm || pattern_name == kPatternDistributedAlpha ||
      pattern_name == kPatternQueryLayerUsePastDistributed || pattern_name == kPatternDistributedAlphaWithLastLayerNorm)
    return true;
  return false;
}

bool EncoderLayerFusion::IsMoe(const std::string &pattern_name) const {
  if (pattern_name == kPatternSigmaQueryLayerDistributed || pattern_name == kPatternSigmaMoeDistributed ||
      pattern_name == kPatternSigmaMoeWithLastLayerNormDistributed || pattern_name == kPatternSigmaMoe ||
      pattern_name == kPatternSigmaMoeWithLastLayerNorm || pattern_name == kPatternSigmaQueryLayerMoe)
    return true;
  return false;
}

AnfNodePtr EncoderLayerFusion::Process(const std::string &pattern_name, const mindspore::FuncGraphPtr &func_graph,
                                       const mindspore::AnfNodePtr &node, const mindspore::EquivPtr &equiv) const {
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    return nullptr;
  }
  bool mask = true;
  is_layernorm_ = IsLastLayerNorm(pattern_name);
  is_layernorm_fusion_ = IsLayerNormFusion(pattern_name);
  is_use_past_ = IsUsePast(pattern_name);
  is_moe_ = IsMoe(pattern_name);

  is_position_bias_ = false;
  if (pattern_name == kPatternEncoderLayerT5Pre || pattern_name == kPatternEncoderLayerT5Post ||
      pattern_name == kPatternEncoderLayerNormT5Pre)
    is_position_bias_ = true;

  bool post_layernorm = false;
  if (pattern_name == kPatternEncoderLayerPost || pattern_name == kPatternEncoderLayerPostNorm ||
      pattern_name == kPatternEncoderLayerT5Post)
    post_layernorm = true;

  is_query_layer_ = false;
  if (pattern_name == kPatternQueryLayerUsePast || pattern_name == kPatternQueryLayerUsePastDistributed ||
      pattern_name == kPatternSigmaQueryLayerDistributed || pattern_name == kPatternSigmaQueryLayerMoe)
    is_query_layer_ = true;

  return CreateMaskedEncoderLayerFusionNode(func_graph, equiv, node, post_layernorm, mask);
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

STATUS EncoderLayerFusion::GetEps(const EquivPtr &equiv, VarPtr node_name, float *eps) const {
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
                                        int *head_size, float *eps1, float *eps2, float *eps3, float *scale) const {
  auto attn_input = GetAttribute(func_graph, equiv, is_attention_);
  MS_ASSERT(attn_input != nullptr);
  auto attn_prim = ops::GetOperator<ops::Attention>(attn_input);
  if (attn_prim->GetAttr(ops::kNumHeads) != nullptr) *head_num = attn_prim->get_head_num();
  if (attn_prim->GetAttr(ops::kSizePerHead) != nullptr) *head_size = attn_prim->get_head_size();
  if (attn_prim->GetAttr(ops::kPositionBias1) != nullptr) is_position_bias_ = attn_prim->get_position_bias();
  if (attn_prim->GetAttr(ops::kScale) != nullptr) *scale = attn_prim->get_scale();
  if (is_layernorm_fusion_) {
    auto layrn1_input = GetAttribute(func_graph, equiv, is_layernorm1_);
    auto layrn1_prim = ops::GetOperator<ops::LayerNormFusion>(layrn1_input);
    if (layrn1_prim->GetAttr(ops::kEpsilon) != nullptr) *eps1 = layrn1_prim->get_epsilon();
    auto layrn2_input = GetAttribute(func_graph, equiv, is_layernorm2_);
    auto layrn2_prim = ops::GetOperator<ops::LayerNormFusion>(layrn2_input);
    if (layrn2_prim->GetAttr(ops::kEpsilon) != nullptr) *eps2 = layrn2_prim->get_epsilon();
    if (is_layernorm_) {
      auto layrn3_input = GetAttribute(func_graph, equiv, is_layernorm3_);
      auto layrn3_prim = ops::GetOperator<ops::LayerNormFusion>(layrn3_input);
      if (layrn3_prim->GetAttr(ops::kEpsilon) != nullptr) *eps3 = layrn3_prim->get_epsilon();
    }
  } else {
    if (GetEps(equiv, eps1_, eps1) != RET_OK) {
      MS_LOG(ERROR) << "not found eps1";
      return RET_ERROR;
    }

    if (GetEps(equiv, eps2_, eps2) != RET_OK) {
      MS_LOG(ERROR) << "not found eps2";
      return RET_ERROR;
    }
    if (is_layernorm_) {
      if (GetEps(equiv, eps3_, eps3) != RET_OK) {
        MS_LOG(ERROR) << "not found eps3";
        return RET_ERROR;
      }
    }
  }
  act_type_ = (is_position_bias_) ? ActType::ActType_Relu : ActType::ActType_Gelu;
  if (!is_position_bias_ && !is_use_past_ && !is_query_layer_) {
    if (!IsActGELU(func_graph, equiv, is_act_)) {
      return RET_ERROR;
    }
  }
  return RET_OK;
}

std::shared_ptr<ops::EncoderLayer> EncoderLayerFusion::CreatePrim(const FuncGraphPtr &func_graph, const EquivPtr &equiv,
                                                                  int64_t ffn_hidden_size, int64_t expert_num,
                                                                  int64_t expert_offset, float capacity_factor) const {
  auto encoder_layer_prim = std::make_shared<ops::EncoderLayer>();
  if (encoder_layer_prim == nullptr) {
    MS_LOG(ERROR) << "Build enoder layer primitive failed.";
    return nullptr;
  }
  int head_num = 0;
  int head_size = 0;
  float eps1 = 1e-5;
  float eps2 = 1e-5;
  float eps3 = 1e-5;
  float scale = 1.0f;
  if (CheckPattern(func_graph, equiv, &head_num, &head_size, &eps1, &eps2, &eps3, &scale)) {
    return nullptr;
  }
  encoder_layer_prim->Init(head_num, head_size, eps1, eps2, eps3, ffn_hidden_size, expert_num, expert_offset,
                           capacity_factor, is_position_bias_, is_post_layernorm_, scale, act_type_, is_layernorm_,
                           is_use_past_, is_query_layer_, is_moe_);

  return encoder_layer_prim;
}

void EncoderLayerFusion::InitAttributes(AnfNodePtr input, AnfNodePtr begin_expert_ids, AnfNodePtr weight_m,
                                        AnfNodePtr expert_capacity_node, int *ffn_hidden_size, int *expert_num,
                                        int *expert_offset, float *capacity_factor) const {
  auto base_shape_ptr = weight_m->Shape();
  auto input_shape_ptr = base_shape_ptr->cast<abstract::ShapePtr>();
  auto input_shape = input_shape_ptr->shape();
  if (is_moe_) {
    auto begin_expert_ids_node = begin_expert_ids->cast<ValueNodePtr>();
    *expert_num = (int64_t)input_shape[0];
    *expert_offset = CastToInt(begin_expert_ids_node->value())[0];
    ShapeVector input_shape1;
    FetchShapeFromAbstract(input->abstract(), &input_shape1);
    int seq = input_shape1[1];
    auto expert_capacity_value_node = utils::cast<ValuePtr>(utils::cast<ValueNodePtr>(expert_capacity_node)->value());
    if (expert_capacity_value_node->isa<tensor::Tensor>()) {
      auto tensor = expert_capacity_value_node->cast<tensor::TensorPtr>();
      auto expert_capacity = *(reinterpret_cast<float *>(tensor->data().data()));
      *capacity_factor = expert_capacity * (*expert_num) / seq;
    }
    *ffn_hidden_size = static_cast<int>(input_shape[C2NUM]);
  } else {
    *ffn_hidden_size = static_cast<int>(input_shape[1]);
  }
}

CNodePtr EncoderLayerFusion::CreateMaskedEncoderLayerFusionNode(const FuncGraphPtr &func_graph, const EquivPtr &equiv,
                                                                const AnfNodePtr &node, bool post_layernorm,
                                                                bool mask = true) const {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(equiv != nullptr);
  MS_ASSERT(node != nullptr);
  AnfNodePtr position_bias, input_mask, bias_attn_o, bias_attn_qkv, beta1, beta2, bias_m, bias_p, beta3, gamma3, v_past,
    k_past, init_reset, batch_valid_length, input_q, weight_q, embedding_table, expert_ids, begin_expert_ids,
    expert_capacity_node;
  auto input = utils::cast<AnfNodePtr>((*equiv)[input_]);
  if (is_moe_) {
    expert_ids = utils::cast<AnfNodePtr>((*equiv)[expert_ids_]);
    begin_expert_ids = utils::cast<AnfNodePtr>((*equiv)[begin_expert_ids_]);
    expert_capacity_node = utils::cast<AnfNodePtr>((*equiv)[expert_capacity_]);
  }
  auto weight_qkv = utils::cast<AnfNodePtr>((*equiv)[weight_attn_qkv_]);
  auto weight_attn_o = utils::cast<AnfNodePtr>((*equiv)[weight_attn_o_]);
  auto weight_m = utils::cast<AnfNodePtr>((*equiv)[weight_m_]);
  auto weight_p = utils::cast<AnfNodePtr>((*equiv)[weight_p_]);
  if (is_query_layer_) {
    init_reset = utils::cast<AnfNodePtr>((*equiv)[init_reset_]);
    batch_valid_length = utils::cast<AnfNodePtr>((*equiv)[batch_valid_length_]);
    input_q = utils::cast<AnfNodePtr>((*equiv)[input_q_]);
    weight_q = utils::cast<AnfNodePtr>((*equiv)[weight_attn_q_]);
    embedding_table = utils::cast<AnfNodePtr>((*equiv)[embedding_table_]);
  }
  if (!is_position_bias_) {
    bias_attn_qkv = utils::cast<AnfNodePtr>((*equiv)[bias_attn_qkv_]);
    bias_attn_o = utils::cast<AnfNodePtr>((*equiv)[bias_attn_o_]);
    bias_m = utils::cast<AnfNodePtr>((*equiv)[bias_m_]);
    bias_p = utils::cast<AnfNodePtr>((*equiv)[bias_p_]);
    beta1 = utils::cast<AnfNodePtr>((*equiv)[beta1_]);
    beta2 = utils::cast<AnfNodePtr>((*equiv)[beta2_]);
    if (is_layernorm_) beta3 = utils::cast<AnfNodePtr>((*equiv)[beta3_]);
  }
  auto gamma1 = utils::cast<AnfNodePtr>((*equiv)[gamma1_]);
  auto gamma2 = utils::cast<AnfNodePtr>((*equiv)[gamma2_]);
  if (is_layernorm_) gamma3 = utils::cast<AnfNodePtr>((*equiv)[gamma3_]);
  input_mask = mask ? utils::cast<AnfNodePtr>((*equiv)[mask_]) : nullptr;
  int ffn_hidden_size;
  int expert_num = 1;
  int expert_offset = 0;
  float capacity_factor = 0;
  InitAttributes(input, begin_expert_ids, weight_m, expert_capacity_node, &ffn_hidden_size, &expert_num, &expert_offset,
                 &capacity_factor);
  auto encoder_layer_prim = CreatePrim(func_graph, equiv, ffn_hidden_size, expert_num, expert_offset, capacity_factor);
  auto encoder_layer_prim_c = encoder_layer_prim->GetPrim();
  MS_CHECK_TRUE_RET(encoder_layer_prim_c != nullptr, nullptr);
  auto value_node = NewValueNode(encoder_layer_prim_c);
  MS_CHECK_TRUE_RET(value_node != nullptr, nullptr);
  std::vector<AnfNodePtr> new_node_inputs = {value_node, input};
  if (is_position_bias_) {
    position_bias = utils::cast<AnfNodePtr>((*equiv)[position_bias_]);
    new_node_inputs.insert(new_node_inputs.end(), {gamma1, weight_qkv});
    if (mask) new_node_inputs.push_back(input_mask);
    new_node_inputs.insert(new_node_inputs.end(), {position_bias, weight_attn_o, gamma2, weight_m, weight_p});
    if (is_layernorm_) new_node_inputs.push_back(gamma3);
  } else {
    if (!is_post_layernorm_) {
      if (is_use_past_) {
        k_past = utils::cast<AnfNodePtr>((*equiv)[k_past_]);
        v_past = utils::cast<AnfNodePtr>((*equiv)[v_past_]);
        new_node_inputs.insert(new_node_inputs.end(), {k_past, v_past});
      }
      if (is_query_layer_)
        new_node_inputs.insert(new_node_inputs.end(), {gamma1, beta1, input_q, weight_q, weight_qkv, bias_attn_qkv});
      else
        new_node_inputs.insert(new_node_inputs.end(), {gamma1, beta1, weight_qkv, bias_attn_qkv});
      if (mask) new_node_inputs.push_back(input_mask);
      new_node_inputs.insert(new_node_inputs.end(), {weight_attn_o, bias_attn_o, gamma2, beta2});
      if (is_moe_) new_node_inputs.push_back(expert_ids);
      new_node_inputs.insert(new_node_inputs.end(), {weight_m, bias_m, weight_p, bias_p});
    } else {
      new_node_inputs.insert(new_node_inputs.end(), {weight_qkv, bias_attn_qkv});
      if (mask) new_node_inputs.push_back(input_mask);
      new_node_inputs.insert(new_node_inputs.end(), {weight_attn_o, bias_attn_o, gamma1, beta1, weight_m, bias_m,
                                                     weight_p, bias_p, gamma2, beta2});
    }
    if (is_layernorm_) new_node_inputs.insert(new_node_inputs.end(), {gamma3, beta3});
  }
  if (is_query_layer_) {
    new_node_inputs.insert(new_node_inputs.end(), {embedding_table, init_reset, batch_valid_length});
  } else if (is_use_past_) {  // temporary solution
    auto inputs = func_graph->get_inputs();
    MS_CHECK_TRUE_RET(inputs.size() > C2NUM, nullptr);
    new_node_inputs.insert(new_node_inputs.end(), {inputs.end()[-2], inputs.end()[-1]});
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
