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

#include "plugin/device/ascend/optimizer/ge/ge_specialized_prepare.h"

#include <vector>
#include <memory>
#include <unordered_map>

#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
void GeTensorArrayPrepare::InsertFlowOutputToTA(const AnfNodePtr &node) {
  auto fg = node->func_graph();
  MS_EXCEPTION_IF_NULL(fg);
  auto mgr = fg->manager();
  MS_EXCEPTION_IF_NULL(mgr);
  CNodePtr new_tuple_get_node = nullptr;
  const auto users = mgr->node_users()[node];
  for (auto &iter : users) {
    auto user_node = iter.first;
    MS_EXCEPTION_IF_NULL(user_node);
    auto ta_node = user_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(ta_node);
    if (new_tuple_get_node == nullptr) {
      new_tuple_get_node = CreatTupleGetItemNode(fg, node, 0);
    }
    ta_node->set_input(iter.second, new_tuple_get_node);
  }
}

void GeTensorArrayPrepare::TransformTASizeFromAttrToInput(const AnfNodePtr &node) {
  auto ta_node = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(ta_node);
  int32_t res_size = 0;
  auto prim = GetValueNode<PrimitivePtr>(ta_node->input(0));
  MS_EXCEPTION_IF_NULL(prim);
  // get size attr
  if (prim->HasAttr("size")) {
    auto size_value_ptr = prim->GetAttr("size");
    auto size = GetValue<int64_t>(size_value_ptr);
    res_size = static_cast<int32_t>(size);
  }
  // generate size input
  auto size_node = NewValueNode(MakeValue(res_size));
  auto node_abstract = std::make_shared<abstract::AbstractScalar>(res_size);
  MS_EXCEPTION_IF_NULL(size_node);
  size_node->set_abstract(node_abstract);
  auto origin_inputs = ta_node->inputs();
  // set cnode input
  ta_node->add_input(size_node);
  // has monad input
  if (origin_inputs.size() > 1) {
    std::vector<AnfNodePtr> sorted_inputs(origin_inputs);
    (void)sorted_inputs.insert(sorted_inputs.cbegin() + 1, size_node);
    ta_node->set_inputs(sorted_inputs);
  }

  // get origin abstract
  auto origin_ta_abstract = ta_node->abstract();
  // new tuple abstract
  std::vector<AbstractBasePtr> abstract_list;
  // push origin abstract
  abstract_list.push_back(origin_ta_abstract);
  // new flow abstract
  float flow_value = 0.0;
  auto flow_abstract = std::make_shared<abstract::AbstractScalar>(flow_value);
  // push flow abstract
  abstract_list.push_back(flow_abstract);

  // modify TensorArray node output's abstract from Tensor to Tuple
  auto new_ta_abstract = std::make_shared<abstract::AbstractTuple>(abstract_list);
  ta_node->set_abstract(new_ta_abstract);
}

const BaseRef GeTensorArrayPrepare::DefinePattern() const {
  VarPtr seq_xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimTensorArray, seq_xs});
}

const AnfNodePtr GeTensorArrayPrepare::Process(const FuncGraphPtr &, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(node);
  TransformTASizeFromAttrToInput(node);
  InsertFlowOutputToTA(node);
  return node;
}
}  // namespace opt
}  // namespace mindspore
