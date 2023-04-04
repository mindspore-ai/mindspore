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
namespace mindspore::opt {
namespace {
const auto &p1 = std::placeholders::_1;
}

const AnfNodePtr UsePastEmbedding::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                           const EquivPtr &equiv) const {
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
  if ((*equiv)[is_gather2_] == nullptr || !utils::isa<AnfNodePtr>((*equiv)[is_gather2_])) {
    MS_LOG(ERROR) << is_gather2_ << "is not AnfNodePtr";
    return nullptr;
  }
  AnfNodePtr gather_node = utils::cast<AnfNodePtr>((*equiv)[is_gather2_]);
  MS_ASSERT(gather_node != nullptr);
  if (gather_node == nullptr || !utils::isa<CNodePtr>(gather_node)) {
    auto users = manager->node_users();
    auto it = users.find(gather_node);
    if (it != users.end()) {
      gather_node = it->second.front().first;
    }
    if (gather_node == nullptr || !utils::isa<CNodePtr>(gather_node)) {
      return nullptr;
    }
  }
  if (!utils::isa<ValueNodePtr>((*equiv)[position_ids_])) {
    MS_LOG(ERROR) << "position_ids_ is not a value node";
    return nullptr;
  }
  auto position_ids = utils::cast<ValueNodePtr>((*equiv)[position_ids_]);
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
  new_node->set_fullname_with_scope("embedding_layer_use_past");
  auto abstract = position_ids->abstract()->Clone();
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "Create tensor abstract failed";
    return nullptr;
  }
  new_node->set_abstract(abstract);
  (void)manager->SetEdge(gather_node, C2NUM, new_node);
  return nullptr;
}

const BaseRef UsePastEmbedding::DefinePattern() const {
  auto input_ids = std::make_shared<Var>("input_ids");
  MS_CHECK_TRUE_RET(input_ids != nullptr, false);
  auto embedding_table_input = std::make_shared<Var>("embedding_table");
  MS_CHECK_TRUE_RET(embedding_table_input != nullptr, false);
  auto var1 = std::make_shared<Var>("var1");
  MS_CHECK_TRUE_RET(var1 != nullptr, {});
  auto is_gather = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimGather), "is_gather");
  MS_CHECK_TRUE_RET(is_gather != nullptr, {});
  auto gather = VectorRef({is_gather, embedding_table_input, input_ids, var1});
  position_ids_ = std::make_shared<Var>("position_ids");
  MS_CHECK_TRUE_RET(position_ids_ != nullptr, false);
  auto embedding_table_pos = std::make_shared<Var>("embedding_table_pos");
  MS_CHECK_TRUE_RET(embedding_table_pos != nullptr, false);
  is_gather2_ = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimGather), "is_gather2");
  MS_CHECK_TRUE_RET(is_gather2_ != nullptr, {});
  auto gather2 = VectorRef({is_gather2_, embedding_table_pos, position_ids_, var1});
  auto is_add = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAddFusion), "is_add");
  MS_CHECK_TRUE_RET(is_add != nullptr, {});
  auto add = VectorRef({is_add, gather, gather2});
  return add;
}
}  // namespace mindspore::opt
