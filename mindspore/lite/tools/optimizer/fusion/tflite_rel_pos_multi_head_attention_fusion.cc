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
#include "tools/optimizer/fusion/tflite_rel_pos_multi_head_attention_fusion.h"
#include <functional>
#include <utility>
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/converter/quant_param_holder.h"
#include "mindspore/core/ops/transpose.h"
#include "nnacl/op_base.h"

namespace mindspore::opt {
namespace {
const auto &p1 = std::placeholders::_1;
const size_t kWeightQueryIndex = 4;
const size_t kWeightKeyIndex = 5;
const size_t kWeightValueIndex = 6;
const size_t kWeightPosIndex = 7;
const size_t kWeightOutputIndex = 10;
const size_t kStackParamSize = 2;
const size_t kInputSize = 16;
const size_t kOutputSize = 2;
}  // namespace
bool TfliteRelPosMultiHeadAttentionFusion::Init() const {
  if (!MultiHeadAttentionFusion::Init()) {
    MS_LOG(ERROR) << "basic class initial member failed.";
    return false;
  }
  query_u_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(query_u_ != nullptr, false);
  query_v_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(query_v_ != nullptr, false);
  input_p_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(input_p_ != nullptr, false);
  weight_p_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(weight_p_ != nullptr, false);
  query_prim_ = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimFullConnection));
  MS_CHECK_TRUE_RET(query_prim_ != nullptr, false);
  key_prim_ = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimFullConnection));
  MS_CHECK_TRUE_RET(key_prim_ != nullptr, false);
  value_prim_ = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimFullConnection));
  MS_CHECK_TRUE_RET(value_prim_ != nullptr, false);
  output_prim_ = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimFullConnection));
  MS_CHECK_TRUE_RET(output_prim_ != nullptr, false);
  pos_prim_ = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimFullConnection));
  MS_CHECK_TRUE_RET(pos_prim_ != nullptr, false);

  for (size_t i = 0; i < kStackParamSize; i++) {
    auto is_var1 = std::make_shared<Var>();
    MS_CHECK_TRUE_RET(is_var1 != nullptr, false);
    query_stack_params_.emplace_back(is_var1);
    auto is_var2 = std::make_shared<Var>();
    MS_CHECK_TRUE_RET(is_var2 != nullptr, false);
    key_stack_params_.emplace_back(is_var2);
    auto is_var3 = std::make_shared<Var>();
    MS_CHECK_TRUE_RET(is_var3 != nullptr, false);
    value_stack_params_.emplace_back(is_var3);
    auto is_var4 = std::make_shared<Var>();
    MS_CHECK_TRUE_RET(is_var4 != nullptr, false);
    pos_stack_params_.emplace_back(is_var4);
  }
  return true;
}

std::unordered_map<std::string, VectorRef> TfliteRelPosMultiHeadAttentionFusion::DefinePatterns() const {
  if (!Init()) {
    MS_LOG(ERROR) << "initial member failed.";
    return {};
  }
  auto query = DefineProcessInputPattern(input_q_, weight_q_, bias_q_, query_stack_params_, query_prim_);
  auto is_add1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAddFusion));
  MS_CHECK_TRUE_RET(is_add1 != nullptr, {});
  auto query_with_bias_u = VectorRef({is_add1, query, query_u_});
  auto is_transpose1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTranspose));
  MS_CHECK_TRUE_RET(is_transpose1 != nullptr, {});
  auto is_param1 = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_param1 != nullptr, {});
  query_with_bias_u = VectorRef({is_transpose1, query_with_bias_u, is_param1});
  auto key = DefineProcessInputPattern(input_k_, weight_k_, bias_k_, key_stack_params_, key_prim_, true);
  MS_CHECK_TRUE_RET(!key.empty(), {});

  auto is_matmul1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMul));
  MS_CHECK_TRUE_RET(is_matmul1 != nullptr, {});
  auto logits_with_u = VectorRef({is_matmul1, query_with_bias_u, key});

  auto is_add2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAddFusion));
  MS_CHECK_TRUE_RET(is_add2 != nullptr, {});
  auto query_with_bias_v = VectorRef({is_add2, query, query_v_});
  auto is_transpose2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTranspose));
  MS_CHECK_TRUE_RET(is_transpose2 != nullptr, {});
  auto is_param2 = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_param2 != nullptr, {});
  query_with_bias_v = VectorRef({is_transpose2, query_with_bias_v, is_param2});

  auto pos = DefineProcessInputPattern(input_p_, weight_p_, nullptr, pos_stack_params_, pos_prim_, true);
  MS_CHECK_TRUE_RET(!pos.empty(), {});
  auto is_transpose3 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTranspose));
  MS_CHECK_TRUE_RET(is_transpose3 != nullptr, {});
  auto is_param3 = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_param3 != nullptr, {});
  pos = VectorRef({is_transpose3, pos, is_param3});
  auto is_matmul2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMul));
  MS_CHECK_TRUE_RET(is_matmul2 != nullptr, {});
  auto logits_with_v = VectorRef({is_matmul2, query_with_bias_v, pos});
  logits_with_v = DefineRelativeShiftPattern(logits_with_v);
  MS_CHECK_TRUE_RET(!logits_with_v.empty(), {});

  auto is_add3 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAddFusion));
  MS_CHECK_TRUE_RET(is_add3 != nullptr, {});
  auto logits = VectorRef({is_add3, logits_with_u, logits_with_v});
  auto is_mul = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMulFusion));
  MS_CHECK_TRUE_RET(is_mul != nullptr, {});
  auto is_param4 = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_param4 != nullptr, {});
  auto logits_div = VectorRef({is_mul, logits, is_param4});
  auto is_add4 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAddFusion));
  MS_CHECK_TRUE_RET(is_add4 != nullptr, {});
  auto is_seq_var = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_seq_var != nullptr, {});
  logits_div = VectorRef({is_add4, logits_div, is_seq_var});
  auto is_softmax = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimSoftmax));
  MS_CHECK_TRUE_RET(is_softmax != nullptr, {});
  auto logits_softmax = VectorRef({is_softmax, logits_div});
  auto value = DefineProcessInputPattern(input_v_, weight_v_, bias_v_, value_stack_params_, value_prim_, true);
  auto is_matmul3 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMul));
  MS_CHECK_TRUE_RET(is_matmul3 != nullptr, {});
  auto output = VectorRef({is_matmul3, logits_softmax, value});
  auto pattern = DefineProcessOutputPattern(output, weight_o_, bias_o_);
  MS_CHECK_TRUE_RET(!pattern.empty(), {});
  std::unordered_map<std::string, VectorRef> patterns;
  patterns.insert(std::make_pair(kRPMHAttentionPatternName, pattern));
  return patterns;
}

AnfNodePtr TfliteRelPosMultiHeadAttentionFusion::Process(const std::string &pattern_name,
                                                         const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                         const EquivPtr &equiv) const {
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    return nullptr;
  }
  return CreateRelPosMultiHeadAttentionNode(func_graph, equiv, node->fullname_with_scope());
}

STATUS GetIntParameterData(const ParameterPtr &param_ptr, int *result) {
  if (!param_ptr->has_default()) {
    MS_LOG(DEBUG) << "param not have default";
    return RET_ERROR;
  }
  auto default_param = param_ptr->default_param();
  if (!utils::isa<tensor::TensorPtr>(default_param)) {
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
    std::accumulate(default_param_ptr->shape().begin(), default_param_ptr->shape().end(), 1, std::multiplies<int>());
  if (shape_size < 1) {
    MS_LOG(ERROR) << "Shape is less than 1.";
    return RET_ERROR;
  }
  *result = ptr[0];
  return RET_OK;
}

std::shared_ptr<ops::Attention> TfliteRelPosMultiHeadAttentionFusion::BuildAttentionPrim(const EquivPtr &equiv) const {
  MS_ASSERT(equiv != nullptr);
  auto attention_prim = std::make_shared<ops::Attention>();
  if (attention_prim == nullptr) {
    MS_LOG(ERROR) << "Build attention primitive failed.";
    return attention_prim;
  }

  std::vector<int> shape_q;
  for (auto &queue_stack_param : query_stack_params_) {
    auto reshape_q = utils::cast<ParameterPtr>((*equiv)[queue_stack_param]);
    int dim;
    if (RET_OK != GetIntParameterData(reshape_q, &dim)) {
      MS_LOG(ERROR) << "Get reshape q data failed";
      return nullptr;
    }
    shape_q.emplace_back(dim);
  }

  std::vector<int> shape_k;
  for (auto &key_stack_param : key_stack_params_) {
    auto reshape_k = utils::cast<ParameterPtr>((*equiv)[key_stack_param]);
    int dim;
    if (RET_OK != GetIntParameterData(reshape_k, &dim)) {
      MS_LOG(ERROR) << "Get reshape k data failed";
      return nullptr;
    }
    shape_k.emplace_back(dim);
  }
  return attention_prim;
}

CNodePtr TfliteRelPosMultiHeadAttentionFusion::CreateRelPosMultiHeadAttentionNode(const FuncGraphPtr &func_graph,
                                                                                  const EquivPtr &equiv,
                                                                                  const std::string &base_name) const {
  MS_ASSERT(func_graph != nullptr && equiv != nullptr);
  auto attention_prim = BuildAttentionPrim(equiv);
  MS_CHECK_TRUE_RET(attention_prim != nullptr, nullptr);
  if (SetQuantParamForAttentionNode(attention_prim, equiv) != lite::RET_OK) {
    MS_LOG(ERROR) << "set quant param for attehtion node failed.";
    return nullptr;
  }
  auto value_node = NewValueNode(attention_prim);
  MS_CHECK_TRUE_RET(value_node != nullptr, nullptr);
  auto input_q = utils::cast<AnfNodePtr>((*equiv)[input_q_]);
  auto input_k = utils::cast<AnfNodePtr>((*equiv)[input_k_]);
  auto input_v = utils::cast<AnfNodePtr>((*equiv)[input_v_]);
  auto input_p = utils::cast<AnfNodePtr>((*equiv)[input_p_]);

  auto weight_q = utils::cast<AnfNodePtr>((*equiv)[weight_q_]);
  auto transpose_prim = std::make_shared<ops::Transpose>();
  MS_CHECK_TRUE_RET(transpose_prim != nullptr, nullptr);
  auto transpose_perm = BuildIntVecParameterNode(func_graph, {1, 0}, "transpose" + base_name + "_perm");
  MS_CHECK_TRUE_RET(transpose_perm != nullptr, nullptr);
  auto weight_q_transpose = func_graph->NewCNode(transpose_prim, {weight_q, transpose_perm});
  MS_CHECK_TRUE_RET(weight_q_transpose != nullptr, nullptr);
  weight_q_transpose->set_fullname_with_scope("transpose_wq" + base_name);

  auto weight_k = utils::cast<AnfNodePtr>((*equiv)[weight_k_]);
  auto weight_k_transpose = func_graph->NewCNode(transpose_prim, {weight_k, transpose_perm});
  MS_CHECK_TRUE_RET(weight_k_transpose != nullptr, nullptr);
  weight_k_transpose->set_fullname_with_scope("transpose_wk" + base_name);

  auto weight_v = utils::cast<AnfNodePtr>((*equiv)[weight_v_]);
  auto weight_v_transpose = func_graph->NewCNode(transpose_prim, {weight_v, transpose_perm});
  MS_CHECK_TRUE_RET(weight_v_transpose != nullptr, nullptr);
  weight_v_transpose->set_fullname_with_scope("transpose_wv" + base_name);

  auto weight_p = utils::cast<AnfNodePtr>((*equiv)[weight_p_]);

  auto weight_o = utils::cast<AnfNodePtr>((*equiv)[weight_o_]);
  auto weight_o_transpose = func_graph->NewCNode(transpose_prim, {weight_o, transpose_perm});
  MS_CHECK_TRUE_RET(weight_o_transpose != nullptr, nullptr);
  weight_o_transpose->set_fullname_with_scope("transpose_wo" + base_name);

  auto bias_q = utils::cast<AnfNodePtr>((*equiv)[bias_q_]);
  auto bias_k = utils::cast<AnfNodePtr>((*equiv)[bias_k_]);
  auto bias_v = utils::cast<AnfNodePtr>((*equiv)[bias_v_]);
  auto bias_o = utils::cast<AnfNodePtr>((*equiv)[bias_o_]);

  auto query_u = utils::cast<AnfNodePtr>((*equiv)[query_u_]);
  auto query_v = utils::cast<AnfNodePtr>((*equiv)[query_v_]);

  std::vector<AnfNodePtr> new_node_inputs = {value_node,
                                             input_q,
                                             input_k,
                                             input_v,
                                             input_p,
                                             weight_q_transpose,
                                             weight_k_transpose,
                                             weight_v_transpose,
                                             weight_p,
                                             query_u,
                                             query_v,
                                             weight_o_transpose,
                                             bias_q,
                                             bias_k,
                                             bias_v,
                                             bias_o};
  auto new_node = func_graph->NewCNode(new_node_inputs);
  MS_CHECK_TRUE_RET(new_node != nullptr, nullptr);
  new_node->set_fullname_with_scope(base_name);
  return new_node;
}

int TfliteRelPosMultiHeadAttentionFusion::SetQuantParamForAttentionNode(const PrimitivePtr &prim,
                                                                        const EquivPtr &equiv) const {
  MS_ASSERT(prim != nullptr && equiv != nullptr);
  auto quant_params_holder = std::make_shared<lite::QuantParamHolder>(kInputSize, kOutputSize);
  MS_CHECK_TRUE_RET(quant_params_holder != nullptr, lite::RET_ERROR);
  auto query_prim = GetValueNode<PrimitivePtr>(utils::cast<AnfNodePtr>((*equiv)[query_prim_]));
  MS_CHECK_TRUE_RET(query_prim != nullptr, lite::RET_ERROR);
  auto query_quant_param_holder = query_prim->GetAttr("quant_params");
  if (query_quant_param_holder != nullptr) {
    auto query_quant_param = query_quant_param_holder->cast<lite::QuantParamHolderPtr>();
    MS_CHECK_TRUE_RET(query_quant_param != nullptr, lite::RET_ERROR);
    if (query_quant_param->get_input_quant_params().size() > 1) {
      quant_params_holder->set_input_quant_param(kWeightQueryIndex, query_quant_param->get_input_quant_params().at(1));
    }
  }
  auto key_prim = GetValueNode<PrimitivePtr>(utils::cast<AnfNodePtr>((*equiv)[key_prim_]));
  MS_CHECK_TRUE_RET(key_prim != nullptr, lite::RET_ERROR);
  auto key_quant_param_holder = key_prim->GetAttr("quant_params");
  if (key_quant_param_holder != nullptr) {
    auto key_quant_param = key_quant_param_holder->cast<lite::QuantParamHolderPtr>();
    MS_CHECK_TRUE_RET(key_quant_param != nullptr, lite::RET_ERROR);
    if (key_quant_param->get_input_quant_params().size() > 1) {
      quant_params_holder->set_input_quant_param(kWeightKeyIndex, key_quant_param->get_input_quant_params().at(1));
    }
  }
  auto value_prim = GetValueNode<PrimitivePtr>(utils::cast<AnfNodePtr>((*equiv)[value_prim_]));
  MS_CHECK_TRUE_RET(value_prim != nullptr, lite::RET_ERROR);
  auto value_quant_param_holder = value_prim->GetAttr("quant_params");
  if (value_quant_param_holder != nullptr) {
    auto value_quant_param = value_quant_param_holder->cast<lite::QuantParamHolderPtr>();
    MS_CHECK_TRUE_RET(value_quant_param != nullptr, lite::RET_ERROR);
    if (value_quant_param->get_input_quant_params().size() > 1) {
      quant_params_holder->set_input_quant_param(kWeightValueIndex, value_quant_param->get_input_quant_params().at(1));
    }
  }

  auto pos_prim = GetValueNode<PrimitivePtr>(utils::cast<AnfNodePtr>((*equiv)[pos_prim_]));
  MS_CHECK_TRUE_RET(pos_prim != nullptr, lite::RET_ERROR);
  auto pos_quant_param_holder = pos_prim->GetAttr("quant_params");
  if (pos_quant_param_holder != nullptr) {
    auto pos_quant_param = pos_quant_param_holder->cast<lite::QuantParamHolderPtr>();
    MS_CHECK_TRUE_RET(pos_quant_param != nullptr, lite::RET_ERROR);
    if (pos_quant_param->get_input_quant_params().size() > 1) {
      quant_params_holder->set_input_quant_param(kWeightPosIndex, pos_quant_param->get_input_quant_params().at(1));
    }
  }

  auto output_prim = GetValueNode<PrimitivePtr>(utils::cast<AnfNodePtr>((*equiv)[output_prim_]));
  MS_CHECK_TRUE_RET(output_prim != nullptr, lite::RET_ERROR);
  auto output_quant_param_holder = output_prim->GetAttr("quant_params");
  if (output_quant_param_holder != nullptr) {
    auto output_quant_param = output_quant_param_holder->cast<lite::QuantParamHolderPtr>();
    MS_CHECK_TRUE_RET(output_quant_param != nullptr, lite::RET_ERROR);
    if (output_quant_param->get_input_quant_params().size() > 1) {
      quant_params_holder->set_input_quant_param(kWeightOutputIndex,
                                                 output_quant_param->get_input_quant_params().at(1));
    }
  }

  prim->AddAttr("quant_params", quant_params_holder);
  return lite::RET_OK;
}

const VectorRef TfliteRelPosMultiHeadAttentionFusion::DefineRelativeShiftPattern(const BaseRef &input) const {
  auto is_pad = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimPadFusion));
  MS_CHECK_TRUE_RET(is_pad != nullptr, {});
  auto is_param = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_param != nullptr, {});
  auto pad = VectorRef({is_pad, input, is_param});
  auto is_reshape1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape));
  MS_CHECK_TRUE_RET(is_reshape1 != nullptr, {});
  auto is_seq_var1 = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_seq_var1 != nullptr, {});
  auto reshape1 = VectorRef({is_reshape1, pad, is_seq_var1});
  auto is_strided_slice1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimStridedSlice));
  MS_CHECK_TRUE_RET(is_strided_slice1 != nullptr, {});
  auto is_seq_var2 = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_seq_var2 != nullptr, {});
  auto stride_slice1 = VectorRef({is_strided_slice1, reshape1, is_seq_var2});
  auto is_reshape2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape));
  MS_CHECK_TRUE_RET(is_reshape2 != nullptr, {});
  auto is_seq_var3 = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_seq_var3 != nullptr, {});
  auto reshape2 = VectorRef({is_reshape2, stride_slice1, is_seq_var3});
  auto is_strided_slice2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimStridedSlice));
  MS_CHECK_TRUE_RET(is_strided_slice2 != nullptr, {});
  auto is_seq_var4 = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_seq_var4 != nullptr, {});
  auto stride_slice2 = VectorRef({is_strided_slice2, reshape2, is_seq_var4});
  return stride_slice2;
}

const VectorRef TfliteRelPosMultiHeadAttentionFusion::DefineProcessInputPattern(
  const BaseRef &input, const BaseRef &weight, const BaseRef &bias, const std::vector<VarPtr> &stack_params,
  const VarPtr &full_connect_prim, bool transpose) const {
  MS_ASSERT(full_connect_prim != nullptr);
  auto is_reshape1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape));
  MS_CHECK_TRUE_RET(is_reshape1 != nullptr, {});
  auto is_seq_var1 = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_seq_var1 != nullptr, {});
  auto reshape = VectorRef({is_reshape1, input, is_seq_var1});
  auto result = VectorRef({full_connect_prim, reshape, weight});

  if (bias != nullptr) {
    auto is_reshape2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape));
    MS_CHECK_TRUE_RET(is_reshape2 != nullptr, {});
    auto is_seq_var2 = std::make_shared<SeqVar>();
    MS_CHECK_TRUE_RET(is_seq_var2 != nullptr, {});
    result = VectorRef({is_reshape2, result, is_seq_var2});
    auto is_add = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAddFusion));
    MS_CHECK_TRUE_RET(is_add != nullptr, {});
    result = VectorRef({is_add, result, bias});
  }

  MS_CHECK_TRUE_RET(stack_params.size() == kStackParamSize, {});
  auto is_stack = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimStack));
  MS_CHECK_TRUE_RET(is_stack != nullptr, {});
  auto is_var = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var != nullptr, {});
  auto stack = VectorRef({is_stack, std::make_shared<Var>(), is_var, stack_params.at(0), stack_params.at(1)});
  auto is_reshape3 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape));
  MS_CHECK_TRUE_RET(is_reshape3 != nullptr, {});
  result = VectorRef({is_reshape3, result, stack});
  if (transpose) {
    auto is_transpose = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTranspose));
    MS_CHECK_TRUE_RET(is_transpose != nullptr, {});
    auto is_param = std::make_shared<CondVar>(IsParamNode);
    MS_CHECK_TRUE_RET(is_param != nullptr, {});
    result = VectorRef({is_transpose, result, is_param});
  }
  return result;
}

const VectorRef TfliteRelPosMultiHeadAttentionFusion::DefineProcessOutputPattern(const BaseRef &input,
                                                                                 const BaseRef &weight,
                                                                                 const BaseRef &bias) const {
  auto is_transpose = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTranspose));
  MS_CHECK_TRUE_RET(is_transpose != nullptr, {});
  auto is_seq_var1 = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_seq_var1 != nullptr, {});
  auto transpose = VectorRef({is_transpose, input, is_seq_var1});
  auto is_reshape1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape));
  MS_CHECK_TRUE_RET(is_reshape1 != nullptr, {});
  auto is_seq_var2 = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_seq_var2 != nullptr, {});
  auto reshape = VectorRef({is_reshape1, transpose, is_seq_var2});
  auto result = VectorRef({output_prim_, reshape, weight});
  auto is_reshape2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape));
  MS_CHECK_TRUE_RET(is_reshape2 != nullptr, {});
  auto is_seq_var3 = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_seq_var3 != nullptr, {});
  result = VectorRef({is_reshape2, result, is_seq_var3});
  auto is_add = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAddFusion));
  MS_CHECK_TRUE_RET(is_add != nullptr, {});
  result = VectorRef({is_add, result, bias});
  return result;
}
}  // namespace mindspore::opt
