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
#include "tools/optimizer/fisson/use_past_embedding.h"
#include <memory>
#include <vector>
#include "ops/op_utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/lite_exporter/fetch_content.h"
#include "nnacl/op_base.h"
#include "ops/use_past_embedding.h"
#include "tools/common/tensor_util.h"
#include "ir/manager.h"
#include "tools/optimizer/common/helper.h"
#include "ops/op_name.h"
namespace mindspore::opt {
namespace {
const auto &p1 = std::placeholders::_1;
}

bool isQueryOp(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    auto anf_node = utils::cast<AnfNodePtr>(n);
    if (anf_node->isa<ValueNode>()) {
      auto prim = GetValuePtr<Primitive>(anf_node);
      if (prim->name() == "EncoderLayer") {
        return GetValue<bool>(prim->GetAttr("query_layer")) == true;
      }
    }
  }
  return false;
}
AnfNodePtr GetNode(FuncGraphManagerPtr manager, VarPtr node_name, const EquivPtr &equiv) {
  if ((*equiv)[node_name] == nullptr || !utils::isa<AnfNodePtr>((*equiv)[node_name])) {
    MS_LOG(ERROR) << node_name << "is not AnfNodePtr";
    return nullptr;
  }
  auto node = utils::cast<AnfNodePtr>((*equiv)[node_name]);
  MS_ASSERT(node != nullptr);
  if (node == nullptr || !utils::isa<CNodePtr>(node)) {
    auto users = manager->node_users();
    auto it = users.find(node);
    if (it != users.end()) {
      node = it->second.front().first;
    }
    if (node == nullptr || !utils::isa<CNodePtr>(node)) {
      return nullptr;
    }
  }
  return node;
}

AnfNodePtr UsePastEmbedding::Process(const std::string &pattern_name, const FuncGraphPtr &func_graph,
                                     const AnfNodePtr &node, const EquivPtr &equiv) const {
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    return nullptr;
  }
  if (func_graph == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    MS_LOG(ERROR) << "func_graph is nullptr.";
    return nullptr;
  }
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr.";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }
  AnfNodePtr gather_node;
  ValueNodePtr position_ids;
  if (pattern_name == kEncoderUsePastEmbedding) {
    gather_node = GetNode(manager, is_gather2_, equiv);
    MS_ASSERT(gather_node != nullptr);
    if (!utils::isa<ValueNodePtr>((*equiv)[position_ids_])) {
      MS_LOG(ERROR) << "position_ids_ is not a value node";
      return nullptr;
    }
    position_ids = utils::cast<ValueNodePtr>((*equiv)[position_ids_]);
    MS_ASSERT(position_ids != nullptr);
  } else if (pattern_name == kQueryUsePastEmbedding) {
    gather_node = GetNode(manager, is_gather_query_, equiv);
    MS_ASSERT(gather_node != nullptr);
    if (!utils::isa<ValueNodePtr>((*equiv)[position_ids_query_])) {
      MS_LOG(ERROR) << "position_ids_ is not a value node";
      return nullptr;
    }
    position_ids = utils::cast<ValueNodePtr>((*equiv)[position_ids_query_]);
    MS_ASSERT(position_ids != nullptr);
  }
  auto model_inputs = func_graph->get_inputs();
  MS_CHECK_TRUE_RET(model_inputs.size() > C2NUM, nullptr);
  auto embeddng_prim = std::make_shared<ops::UsePastEmbedding>();
  auto embeddng_prim_c = embeddng_prim->GetPrim();
  MS_ASSERT(embeddng_prim_c != nullptr);
  auto is_embedding = NewValueNode(embeddng_prim_c);
  std::vector<AnfNodePtr> new_node_inputs = {is_embedding, position_ids, model_inputs.end()[-2],
                                             model_inputs.end()[-1]};
  auto new_node = func_graph->NewCNode(new_node_inputs);
  MS_CHECK_TRUE_RET(new_node != nullptr, nullptr);
  new_node->set_fullname_with_scope("embedding_layer_use_past_" + pattern_name);
  auto abstract = position_ids->abstract()->Clone();
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "Create tensor abstract failed";
    return nullptr;
  }
  new_node->set_abstract(abstract);
  (void)manager->SetEdge(gather_node, C2NUM, new_node);
  return nullptr;
}

VectorRef UsePastEmbedding::DefinePatternEncoderEmbedding() const {
  auto input_ids = std::make_shared<Var>("input_ids");
  MS_CHECK_TRUE_RET(input_ids != nullptr, {});
  auto embedding_table_input = std::make_shared<Var>("embedding_table");
  MS_CHECK_TRUE_RET(embedding_table_input != nullptr, {});
  auto var1 = std::make_shared<Var>("var1");
  MS_CHECK_TRUE_RET(var1 != nullptr, {});
  auto is_gather = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimGather), "is_gather");
  MS_CHECK_TRUE_RET(is_gather != nullptr, {});
  auto gather = VectorRef({is_gather, embedding_table_input, input_ids, var1});
  position_ids_ = std::make_shared<Var>("position_ids");
  MS_CHECK_TRUE_RET(position_ids_ != nullptr, {});
  auto embedding_table_pos = std::make_shared<Var>("embedding_table_pos");
  MS_CHECK_TRUE_RET(embedding_table_pos != nullptr, {});
  is_gather2_ = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimGather), "is_gather2");
  MS_CHECK_TRUE_RET(is_gather2_ != nullptr, {});
  auto gather2 = VectorRef({is_gather2_, embedding_table_pos, position_ids_, var1});
  auto is_add = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAddFusion), "is_add");
  MS_CHECK_TRUE_RET(is_add != nullptr, {});
  auto add = VectorRef({is_add, gather, gather2});
  return add;
}

VectorRef UsePastEmbedding::DefinePatternQueryEmbedding() const {
  auto embedding_table_input = std::make_shared<Var>("embedding_table");
  MS_CHECK_TRUE_RET(embedding_table_input != nullptr, {});
  auto var1 = std::make_shared<Var>("var1");
  MS_CHECK_TRUE_RET(var1 != nullptr, {});
  is_gather_query_ = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimGather), "is_gather");
  MS_CHECK_TRUE_RET(is_gather_query_ != nullptr, {});
  position_ids_query_ = std::make_shared<Var>("position_ids");
  MS_CHECK_TRUE_RET(position_ids_query_ != nullptr, {});
  auto gather = VectorRef({is_gather_query_, embedding_table_input, position_ids_query_, var1});
  auto is_reshape1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "reshape-encoder");
  MS_CHECK_TRUE_RET(is_reshape1 != nullptr, {});
  auto var_reshape = std::make_shared<Var>("var_reshape");
  MS_CHECK_TRUE_RET(var_reshape != nullptr, {});
  auto reshape = VectorRef({is_reshape1, gather, var_reshape});
  is_query_ = std::make_shared<CondVar>(isQueryOp);
  MS_CHECK_TRUE_RET(is_query_ != nullptr, {});
  auto input = std::make_shared<Var>("input");
  MS_CHECK_TRUE_RET(input != nullptr, {});
  auto k_past = std::make_shared<Var>("k_past");
  MS_CHECK_TRUE_RET(k_past != nullptr, {});
  auto v_past = std::make_shared<Var>("v_past");
  MS_CHECK_TRUE_RET(v_past != nullptr, {});
  auto gamma1 = std::make_shared<Var>("gamma1");
  MS_CHECK_TRUE_RET(gamma1 != nullptr, {});
  auto beta1 = std::make_shared<Var>("beta1");
  MS_CHECK_TRUE_RET(beta1 != nullptr, {});
  auto input_q = std::make_shared<Var>("input_q");
  MS_CHECK_TRUE_RET(input_q != nullptr, {});
  auto weight_q = std::make_shared<Var>("weight_q");
  MS_CHECK_TRUE_RET(weight_q != nullptr, {});
  auto weight_kv = std::make_shared<Var>("weight_kv");
  MS_CHECK_TRUE_RET(weight_kv != nullptr, {});
  auto mask = std::make_shared<Var>("mask");
  MS_CHECK_TRUE_RET(mask != nullptr, {});
  auto weight_attn_p = std::make_shared<Var>("weight_attn_p");
  MS_CHECK_TRUE_RET(weight_attn_p != nullptr, {});
  auto bias_attn_p = std::make_shared<Var>("bias_attn_p");
  MS_CHECK_TRUE_RET(bias_attn_p != nullptr, {});
  auto gamma2 = std::make_shared<Var>("gamma2");
  MS_CHECK_TRUE_RET(gamma2 != nullptr, {});
  auto beta2 = std::make_shared<Var>("beta2");
  MS_CHECK_TRUE_RET(beta2 != nullptr, {});
  auto expert_ids = std::make_shared<Var>("expert_ids");
  MS_CHECK_TRUE_RET(expert_ids != nullptr, {});
  auto weight_m = std::make_shared<Var>("weight_m");
  MS_CHECK_TRUE_RET(weight_m != nullptr, {});
  auto bias_m = std::make_shared<Var>("bias_m");
  MS_CHECK_TRUE_RET(bias_m != nullptr, {});
  auto weight_p = std::make_shared<Var>("weight_p");
  MS_CHECK_TRUE_RET(weight_p != nullptr, {});
  auto bias_p = std::make_shared<Var>("bias_p");
  MS_CHECK_TRUE_RET(bias_p != nullptr, {});
  auto embedding_table = std::make_shared<Var>("embedding_table");
  MS_CHECK_TRUE_RET(embedding_table != nullptr, {});
  auto init_reset = std::make_shared<Var>("init_reset");
  MS_CHECK_TRUE_RET(init_reset != nullptr, {});
  auto valid_length = std::make_shared<Var>("valid_length");
  MS_CHECK_TRUE_RET(valid_length != nullptr, {});
  auto query_layer =
    VectorRef({is_query_, input,     k_past,   v_past,        gamma1,          beta1,      reshape,     input_q,
               weight_q,  weight_kv, mask,     weight_attn_p, bias_attn_p,     gamma2,     beta2,       expert_ids,
               weight_m,  bias_m,    weight_p, bias_p,        embedding_table, init_reset, valid_length});
  return query_layer;
}

std::unordered_map<std::string, VectorRef> UsePastEmbedding::DefinePatterns() const {
  std::unordered_map<std::string, VectorRef> patterns;
  patterns[kEncoderUsePastEmbedding] = DefinePatternEncoderEmbedding();
  patterns[kQueryUsePastEmbedding] = DefinePatternQueryEmbedding();
  return patterns;
}
}  // namespace mindspore::opt
