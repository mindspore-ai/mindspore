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

namespace mindspore::opt {
namespace {
const auto &p1 = std::placeholders::_1;
}  // namespace

TfliteRelPosMultiHeadAttentionFusion::TfliteRelPosMultiHeadAttentionFusion(const string &name, bool multigraph)
    : MultiHeadAttentionFusion(name, multigraph) {
  query_u_ = std::make_shared<Var>();
  query_v_ = std::make_shared<Var>();
  input_p_ = std::make_shared<Var>();
  weight_p_ = std::make_shared<Var>();
  query_prim_ = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimFullConnection));
  key_prim_ = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimFullConnection));
  value_prim_ = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimFullConnection));
  output_prim_ = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimFullConnection));
  pos_prim_ = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimFullConnection));

  for (size_t i = 0; i < 2; i++) {
    query_stack_params_.emplace_back(std::make_shared<Var>());
    key_stack_params_.emplace_back(std::make_shared<Var>());
    value_stack_params_.emplace_back(std::make_shared<Var>());
    pos_stack_params_.emplace_back(std::make_shared<Var>());
  }
}

std::unordered_map<std::string, VectorRef> TfliteRelPosMultiHeadAttentionFusion::DefinePatterns() const {
  auto query = DefineProcessInputPattern(input_q_, weight_q_, bias_q_, query_stack_params_, query_prim_);
  auto query_with_bias_u =
    VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAddFusion)), query, query_u_});
  query_with_bias_u = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTranspose)),
                                 query_with_bias_u, std::make_shared<CondVar>(IsParamNode)});
  auto key = DefineProcessInputPattern(input_k_, weight_k_, bias_k_, key_stack_params_, key_prim_, true);

  auto logits_with_u =
    VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMul)), query_with_bias_u, key});

  auto query_with_bias_v =
    VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAddFusion)), query, query_v_});
  query_with_bias_v = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTranspose)),
                                 query_with_bias_v, std::make_shared<CondVar>(IsParamNode)});

  auto pos = DefineProcessInputPattern(input_p_, weight_p_, nullptr, pos_stack_params_, pos_prim_, true);
  pos = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTranspose)), pos,
                   std::make_shared<CondVar>(IsParamNode)});
  auto logits_with_v =
    VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMul)), query_with_bias_v, pos});
  logits_with_v = DefineRelativeShiftPattern(logits_with_v);

  auto logits =
    VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAddFusion)), logits_with_u, logits_with_v});
  auto logits_div = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMulFusion)), logits,
                               std::make_shared<CondVar>(IsParamNode)});
  logits_div = VectorRef(
    {std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAddFusion)), logits_div, std::make_shared<SeqVar>()});
  auto logits_softmax = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimSoftmax)), logits_div});
  auto value = DefineProcessInputPattern(input_v_, weight_v_, bias_v_, value_stack_params_, value_prim_, true);
  auto output =
    VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMul)), logits_softmax, value});
  auto pattern = DefineProcessOutputPattern(output, weight_o_, bias_o_);
  std::unordered_map<std::string, VectorRef> patterns;
  patterns.insert(std::make_pair(kRPMHAttentionPatternName, pattern));
  return patterns;
}

AnfNodePtr TfliteRelPosMultiHeadAttentionFusion::Process(const std::string &pattern_name,
                                                         const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                         const EquivPtr &equiv) const {
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
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(equiv != nullptr);
  auto attention_prim = BuildAttentionPrim(equiv);
  if (attention_prim == nullptr) {
    MS_LOG(ERROR) << "Build attention primitive failed.";
    return nullptr;
  }
  auto quant_params_holder = std::make_shared<lite::QuantParamHolder>(16, 1);
  auto query_prim = GetValueNode<PrimitivePtr>(utils::cast<AnfNodePtr>((*equiv)[query_prim_]));
  auto query_quant_param_holder = query_prim->GetAttr("quant_params");
  if (query_quant_param_holder != nullptr) {
    quant_params_holder->set_input_quant_param(
      4, query_quant_param_holder->cast<lite::QuantParamHolderPtr>()->get_input_quant_params().at(1));
  }
  auto key_prim = GetValueNode<PrimitivePtr>(utils::cast<AnfNodePtr>((*equiv)[key_prim_]));
  auto key_quant_param_holder = key_prim->GetAttr("quant_params");
  if (key_quant_param_holder != nullptr) {
    quant_params_holder->set_input_quant_param(
      5, key_quant_param_holder->cast<lite::QuantParamHolderPtr>()->get_input_quant_params().at(1));
  }
  auto value_prim = GetValueNode<PrimitivePtr>(utils::cast<AnfNodePtr>((*equiv)[value_prim_]));
  auto value_quant_param_holder = value_prim->GetAttr("quant_params");
  if (value_quant_param_holder != nullptr) {
    quant_params_holder->set_input_quant_param(
      6, value_quant_param_holder->cast<lite::QuantParamHolderPtr>()->get_input_quant_params().at(1));
  }

  auto pos_prim = GetValueNode<PrimitivePtr>(utils::cast<AnfNodePtr>((*equiv)[pos_prim_]));
  auto pos_quant_param_holder = pos_prim->GetAttr("quant_params");
  if (pos_quant_param_holder != nullptr) {
    quant_params_holder->set_input_quant_param(
      7, pos_quant_param_holder->cast<lite::QuantParamHolderPtr>()->get_input_quant_params().at(1));
  }

  auto output_prim = GetValueNode<PrimitivePtr>(utils::cast<AnfNodePtr>((*equiv)[output_prim_]));
  auto output_quant_param_holder = output_prim->GetAttr("quant_params");
  if (output_quant_param_holder != nullptr) {
    quant_params_holder->set_input_quant_param(
      10, output_quant_param_holder->cast<lite::QuantParamHolderPtr>()->get_input_quant_params().at(1));
  }

  attention_prim->AddAttr("quant_params", quant_params_holder);
  auto value_node = NewValueNode(attention_prim);
  auto input_q = utils::cast<AnfNodePtr>((*equiv)[input_q_]);
  auto input_k = utils::cast<AnfNodePtr>((*equiv)[input_k_]);
  auto input_v = utils::cast<AnfNodePtr>((*equiv)[input_v_]);
  auto input_p = utils::cast<AnfNodePtr>((*equiv)[input_p_]);

  auto weight_q = utils::cast<AnfNodePtr>((*equiv)[weight_q_]);
  auto transpose_prim = std::make_shared<ops::Transpose>();
  auto transpose_perm = BuildIntVecParameterNode(func_graph, {1, 0}, "transpose" + base_name + "_perm");
  auto weight_q_transpose = func_graph->NewCNode(transpose_prim, {weight_q, transpose_perm});
  weight_q_transpose->set_fullname_with_scope("transpose_wq" + base_name);

  auto weight_k = utils::cast<AnfNodePtr>((*equiv)[weight_k_]);
  auto weight_k_transpose = func_graph->NewCNode(transpose_prim, {weight_k, transpose_perm});
  weight_k_transpose->set_fullname_with_scope("transpose_wk" + base_name);

  auto weight_v = utils::cast<AnfNodePtr>((*equiv)[weight_v_]);
  auto weight_v_transpose = func_graph->NewCNode(transpose_prim, {weight_v, transpose_perm});
  weight_v_transpose->set_fullname_with_scope("transpose_wv" + base_name);

  auto weight_p = utils::cast<AnfNodePtr>((*equiv)[weight_p_]);

  auto weight_o = utils::cast<AnfNodePtr>((*equiv)[weight_o_]);
  auto weight_o_transpose = func_graph->NewCNode(transpose_prim, {weight_o, transpose_perm});
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
  new_node->set_fullname_with_scope(base_name);
  return new_node;
}
const VectorRef TfliteRelPosMultiHeadAttentionFusion::DefineRelativeShiftPattern(const BaseRef &input) const {
  auto pad = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimPadFusion)), input,
                        std::make_shared<CondVar>(IsParamNode)});
  auto reshape1 = VectorRef(
    {std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape)), pad, std::make_shared<SeqVar>()});
  auto stride_slice1 = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimStridedSlice)), reshape1,
                                  std::make_shared<SeqVar>()});
  auto reshape2 = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape)), stride_slice1,
                             std::make_shared<SeqVar>()});
  auto stride_slice2 = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimStridedSlice)), reshape2,
                                  std::make_shared<SeqVar>()});
  return stride_slice2;
}

const VectorRef TfliteRelPosMultiHeadAttentionFusion::DefineProcessInputPattern(
  const BaseRef &input, const BaseRef &weight, const BaseRef &bias, const std::vector<VarPtr> &stack_params,
  const VarPtr &full_connect_prim, bool transpose) const {
  auto reshape = VectorRef(
    {std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape)), input, std::make_shared<SeqVar>()});
  auto result = VectorRef({full_connect_prim, reshape, weight});

  if (bias != nullptr) {
    result = VectorRef(
      {std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape)), result, std::make_shared<SeqVar>()});
    result = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAddFusion)), result, bias});
  }

  MS_ASSERT(stack_params.size() == 2);
  auto stack = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimStack)), std::make_shared<Var>(),
                          std::make_shared<Var>(), stack_params.at(0), stack_params.at(1)});
  result = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape)), result, stack});
  if (transpose) {
    result = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTranspose)), result,
                        std::make_shared<CondVar>(IsParamNode)});
  }
  return result;
}

const VectorRef TfliteRelPosMultiHeadAttentionFusion::DefineProcessOutputPattern(const BaseRef &input,
                                                                                 const BaseRef &weight,
                                                                                 const BaseRef &bias) const {
  auto transpose = VectorRef(
    {std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTranspose)), input, std::make_shared<SeqVar>()});
  auto reshape = VectorRef(
    {std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape)), transpose, std::make_shared<SeqVar>()});
  auto result = VectorRef({output_prim_, reshape, weight});
  result = VectorRef(
    {std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape)), result, std::make_shared<SeqVar>()});
  result = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAddFusion)), result, bias});
  return result;
}
}  // namespace mindspore::opt
