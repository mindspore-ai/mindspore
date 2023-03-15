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
#include "plugin/device/cpu/optimizer/insert_cast_to_pyexecute.h"
#include <memory>
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace opt {
const BaseRef InsertCastToPyExecute::DefinePattern() const {
  VarPtr xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimPyExecute, xs});
}

const AnfNodePtr InsertCastToPyExecute::Process(const FuncGraphPtr &fg, const AnfNodePtr &node,
                                                const EquivPtr &) const {
  auto cnode = node->cast<CNodePtr>();
  if (!common::AnfAlgo::HasNodeAttr(kAttrNeedCast, cnode)) {
    return nullptr;
  }
  common::AnfAlgo::EraseNodeAttr(kAttrNeedCast, node);

  auto cast_node =
    insert_cast_function_(fg, node, kOpFormat_DEFAULT, kNumberTypeFloat64, kNumberTypeFloat64, node->Shape());
  common::AnfAlgo::SetNodeAttr(kAttrAnyTypeCast, MakeValue(True), cast_node);
  if (fg->isa<session::KernelGraph>()) {
    auto kg = fg->cast_ptr<session::KernelGraph>();
    kg->ReplaceInternalOutput(node, cast_node);
  }
  return cast_node;
}
}  // namespace opt
}  // namespace mindspore
