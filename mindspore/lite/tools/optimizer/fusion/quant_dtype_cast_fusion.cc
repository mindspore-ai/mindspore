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

#include "tools/optimizer/fusion/quant_dtype_cast_fusion.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace opt {
const BaseRef QuantDtypeCastFusion::DefinePattern() const {
  input_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(input_ != nullptr, {});
  auto is_dtype_cast_first = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimQuantDTypeCast>);
  MS_CHECK_TRUE_RET(is_dtype_cast_first != nullptr, {});
  VectorRef first_cast_ref = VectorRef({is_dtype_cast_first, input_});

  auto is_dtype_cast_second = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimQuantDTypeCast>);
  MS_CHECK_TRUE_RET(is_dtype_cast_second != nullptr, {});
  return VectorRef({is_dtype_cast_second, first_cast_ref});
}

const AnfNodePtr QuantDtypeCastFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                               const EquivPtr &equiv) const {
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    MS_LOG(ERROR) << "input param is nullptr, do norm fusion failed.";
    return nullptr;
  }
  if (!utils::isa<CNodePtr>(node)) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  if (IsMarkedTrainOp(cnode)) {
    return nullptr;
  }
  if (!CheckPattern(func_graph, equiv, node)) {
    return nullptr;
  }

  auto manager = func_graph->manager();
  if (manager == nullptr) {
    manager = Manage(func_graph, true);
  }
  MS_CHECK_TRUE_RET(manager != nullptr, nullptr);
  auto node_users = manager->node_users()[cnode];
  auto input_node = utils::cast<AnfNodePtr>((*equiv)[input_]);
  for (auto &node_user : node_users) {
    manager->SetEdge(node_user.first, node_user.second, input_node);
  }
  return cnode;
}

bool QuantDtypeCastFusion::CheckPattern(const FuncGraphPtr &func_graph, const EquivPtr &equiv,
                                        const AnfNodePtr &node) const {
  auto cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(cnode != nullptr, false);
  auto first_cast_cnode = cnode->input(kInputIndexOne)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(first_cast_cnode != nullptr, false);
  int first_src_dtype = kTypeUnknown;
  int first_dst_dtype = kTypeUnknown;
  int second_dst_dtype = kTypeUnknown;

  // first cast node
  auto first_cast_primitive = GetValueNode<PrimitivePtr>(first_cast_cnode->input(0));
  auto src_dtype_value = first_cast_primitive->GetAttr("src_t");
  MS_CHECK_TRUE_RET(src_dtype_value != nullptr, false);
  auto dst_dtype_value = first_cast_primitive->GetAttr("dst_t");
  MS_CHECK_TRUE_RET(dst_dtype_value != nullptr, false);
  first_src_dtype = GetValue<int64_t>(src_dtype_value);
  first_dst_dtype = GetValue<int64_t>(dst_dtype_value);

  // second cast node
  auto second_cast_primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  dst_dtype_value = second_cast_primitive->GetAttr("dst_t");
  MS_CHECK_TRUE_RET(dst_dtype_value != nullptr, false);
  second_dst_dtype = GetValue<int64_t>(dst_dtype_value);

  bool check_dtype_matched = (first_src_dtype != kTypeUnknown && first_dst_dtype != kTypeUnknown &&
                              second_dst_dtype != kTypeUnknown && first_src_dtype == second_dst_dtype);
  return check_dtype_matched;
}
}  // namespace opt
}  // namespace mindspore
