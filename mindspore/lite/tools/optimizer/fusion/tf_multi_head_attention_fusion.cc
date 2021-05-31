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
#include "tools/optimizer/fusion/tf_multi_head_attention_fusion.h"
#include <functional>
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore::opt {
namespace {
const auto &p1 = std::placeholders::_1;
}  // namespace

TfMultiHeadAttentionFusion::TfMultiHeadAttentionFusion(const string &name, bool multigraph)
    : PatternProcessPass(name, multigraph) {
  input_q_ = std::make_shared<Var>();
  input_k_ = std::make_shared<Var>();
  input_v_ = std::make_shared<Var>();

  weight_q_ = std::make_shared<Var>();
  weight_k_ = std::make_shared<Var>();
  weight_v_ = std::make_shared<Var>();
  weight_o_ = std::make_shared<Var>();

  bias_q_ = std::make_shared<Var>();
  bias_k_ = std::make_shared<Var>();
  bias_v_ = std::make_shared<Var>();
  bias_o_ = std::make_shared<Var>();

  reshape_k_ = std::make_shared<Var>();
  reshape_v_ = std::make_shared<Var>();
}

const BaseRef TfMultiHeadAttentionFusion::DefinePattern() const {
  auto query = DefineProcessInputPattern(input_q_, weight_q_, bias_q_, std::make_shared<CondVar>(IsParamNode));
  auto query_div = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimDivFusion)), query,
                              std::make_shared<CondVar>(IsParamNode)});

  auto key = DefineProcessInputPattern(input_k_, weight_k_, bias_k_, reshape_k_);
  auto query_mul_key =
    VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMul)), query_div, key});
  auto softmax = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimSoftmax)), query_mul_key});

  auto value = DefineProcessInputPattern(input_v_, weight_v_, bias_v_, reshape_v_);
  auto softmax_mul_val =
    VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMul)), softmax, value});

  return DefineProcessOutputPattern(softmax_mul_val, weight_o_, bias_o_);
}

const AnfNodePtr TfMultiHeadAttentionFusion::Process(const mindspore::FuncGraphPtr &func_graph,
                                                     const mindspore::AnfNodePtr &node,
                                                     const mindspore::EquivPtr &equiv) const {
  return CreateMultiHeadAttentionNode(func_graph, equiv, node->fullname_with_scope(), 0);
}

const VectorRef TfMultiHeadAttentionFusion::DefineDensePattern(const BaseRef &input, const BaseRef &weight,
                                                               const BaseRef &bias) const {
  auto transpose = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTranspose)), input,
                              std::make_shared<CondVar>(IsParamNode)});
  auto reshape1 = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape)), transpose,
                             std::make_shared<CondVar>(IsParamNode)});
  auto matmul = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMul)), reshape1, weight});
  auto reshape2 = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape)), matmul,
                             std::make_shared<CondVar>(IsParamNode)});
  if (bias == nullptr) {
    return reshape2;
  }
  auto bias_add = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimBiasAdd)), reshape2, bias});
  return bias_add;
}

const VectorRef TfMultiHeadAttentionFusion::DefineProcessInputPattern(const BaseRef &input, const BaseRef &weight,
                                                                      const BaseRef &bias, const BaseRef &reshape_shape,
                                                                      bool transpose) const {
  auto input_after_dense = DefineDensePattern(input, weight, bias);
  auto result = VectorRef(
    {std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape)), input_after_dense, reshape_shape});
  if (transpose) {
    result = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTranspose)), result,
                        std::make_shared<CondVar>(IsParamNode)});
  }
  return result;
}

const VectorRef TfMultiHeadAttentionFusion::DefineProcessOutputPattern(const BaseRef &input, const BaseRef &weight,
                                                                       const BaseRef &bias) const {
  auto transpose = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTranspose)), input,
                              std::make_shared<CondVar>(IsParamNode)});
  auto reshape = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape)), transpose,
                            std::make_shared<CondVar>(IsParamNode)});
  return DefineDensePattern(reshape, weight, bias);
}

CNodePtr TfMultiHeadAttentionFusion::CreateMultiHeadAttentionNode(const FuncGraphPtr &func_graph, const EquivPtr &equiv,
                                                                  const std::string &base_name, int var_offset) const {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(equiv != nullptr);
  auto attention_prim = BuildAttentionPrim(equiv);
  if (attention_prim == nullptr) {
    MS_LOG(ERROR) << "Build attention primitive failed.";
    return nullptr;
  }
  auto value_node = NewValueNode(attention_prim);
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
  new_node->set_fullname_with_scope(base_name);
  return new_node;
}

STATUS GetIntParameterData(const ParameterPtr &param_ptr, std::vector<int> *result) {
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
  for (int64_t i = 0; i < shape_size; i++) {
    result->emplace_back(ptr[i]);
  }
  return RET_OK;
}

std::shared_ptr<ops::Attention> TfMultiHeadAttentionFusion::BuildAttentionPrim(const EquivPtr &equiv) const {
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
  if (shape_k.size() < 2 || shape_v.size() < 2 || shape_k.at(shape_k.size() - 2) != shape_v.at(shape_v.size() - 2)) {
    MS_LOG(ERROR) << "Shape k or shape v is invalid.";
    return nullptr;
  }
  attention_prim->set_num_heads(shape_k.at(shape_k.size() - 2));
  attention_prim->set_key_dim(shape_k.at(shape_k.size() - 1));
  attention_prim->set_value_dim(shape_v.at(shape_v.size() - 1));
  return attention_prim;
}
}  // namespace mindspore::opt
