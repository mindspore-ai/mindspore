/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/ir_fission/addn_fission.h"
#include <memory>
#include <vector>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
AnfNodePtr AddnFission::CreateNewAddn(const FuncGraphPtr &func_graph, const CNodePtr &origin_addn_cnode,
                                      size_t begin_index, size_t offset) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(origin_addn_cnode);
  std::vector<AnfNodePtr> new_addn_inputs{NewValueNode(std::make_shared<Primitive>(prim::kPrimAddN->name()))};
  for (size_t i = begin_index; i < begin_index + offset; ++i) {
    new_addn_inputs.emplace_back(origin_addn_cnode->input(i));
  }
  CNodePtr new_addn = NewCNode(new_addn_inputs, func_graph);
  MS_EXCEPTION_IF_NULL(new_addn);
  new_addn->set_scope(origin_addn_cnode->scope());
  new_addn->set_abstract(origin_addn_cnode->abstract());
  common::AnfAlgo::SetNodeAttr(kAttrN, MakeValue(SizeToLong(offset)), new_addn);
  std::vector<int64_t> dyn_input_sizes{SizeToLong(offset)};
  common::AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue(dyn_input_sizes), new_addn);
  return new_addn;
}

const BaseRef AddnFission::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimAddN, Xs});
}

const AnfNodePtr AddnFission::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  // The real input begins with index 1.
  size_t origin_input_size = cnode->inputs().size() - 1;
  if (origin_input_size <= inputs_divisor_) {
    return nullptr;
  }
  CNodePtr new_cnode = cnode;
  while (origin_input_size > inputs_divisor_) {
    MS_EXCEPTION_IF_NULL(new_cnode);
    std::vector<AnfNodePtr> base_addn_inputs{NewValueNode(std::make_shared<Primitive>(prim::kPrimAddN->name()))};
    size_t cur_input_index = 1;
    // Divide the inputs of addn by inputs_divisor_.
    while ((origin_input_size - cur_input_index + 1) >= inputs_divisor_) {
      base_addn_inputs.push_back(CreateNewAddn(func_graph, new_cnode, cur_input_index, inputs_divisor_));
      cur_input_index += inputs_divisor_;
    }
    for (size_t i = cur_input_index; i <= origin_input_size; i++) {
      base_addn_inputs.emplace_back(new_cnode->input(i));
    }
    CNodePtr base_addn = NewCNode(base_addn_inputs, func_graph);
    MS_EXCEPTION_IF_NULL(base_addn);
    base_addn->set_scope(new_cnode->scope());
    base_addn->set_abstract(new_cnode->abstract());
    common::AnfAlgo::SetNodeAttr(kAttrN, MakeValue(SizeToLong(base_addn_inputs.size() - 1)), base_addn);
    std::vector<int64_t> dyn_input_sizes{SizeToLong(base_addn_inputs.size() - 1)};
    common::AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue(dyn_input_sizes), base_addn);
    new_cnode = base_addn;
    origin_input_size = base_addn->inputs().size() - 1;
  }

  return new_cnode;
}
}  // namespace opt
}  // namespace mindspore
