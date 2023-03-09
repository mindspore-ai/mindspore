/**
 * Copyright  2019-2023 Huawei Technologies Co., Ltd
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

#include "backend/common/pass/add_input_structural_for_py_execute.h"

#include <memory>
#include <vector>

#include "utils/hash_set.h"
#include "backend/common/pass/const_input_to_attr.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
namespace {
ValuePtr SetInputStructuralFromAbstract(const AbstractBasePtr &abs) {
  if (abs->isa<abstract::AbstractSequence>()) {
    auto seq_abs = abs->cast_ptr<abstract::AbstractSequence>();
    std::vector<ValuePtr> structural;
    for (size_t index = 0; index < seq_abs->size(); ++index) {
      (void)structural.emplace_back(SetInputStructuralFromAbstract((*seq_abs)[index]));
    }
    return std::make_shared<ValueTuple>(structural);
  }
  return MakeValue<int64_t>(-1);
}
}  // namespace
const BaseRef AddInputStructuralForPyExecute::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimPyExecute, Xs});
}
const AnfNodePtr AddInputStructuralForPyExecute::Process(const FuncGraphPtr &, const AnfNodePtr &node,
                                                         const EquivPtr &) const {
  if (!common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimPyExecute)) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  if (common::AnfAlgo::HasNodeAttr(kAttrTupleInputStructural, cnode)) {
    return nullptr;
  }
  std::vector<ValuePtr> input_structurals;
  for (size_t index = 1; index < cnode->size(); ++index) {
    auto input_node = cnode->input(index);
    auto abstract = input_node->abstract();
    MS_EXCEPTION_IF_NULL(abstract);
    if (!abstract->isa<abstract::AbstractMonad>()) {
      (void)input_structurals.emplace_back(SetInputStructuralFromAbstract(abstract));
    }
  }
  auto input_structural = std::make_shared<ValueTuple>(input_structurals);
  common::AnfAlgo::SetNodeAttr(kAttrTupleInputStructural, input_structural, cnode);
  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
