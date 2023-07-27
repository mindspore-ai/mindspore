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
#include "mindspore/core/ops/framework_ops.h"
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
  MS_EXCEPTION_IF_NULL(fg);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->abstract() == nullptr || !cnode->abstract()->isa<abstract::AbstractAny>()) {
    return nullptr;
  }

  // If use tensor supposed dtype.
  static const auto use_supposed_dtype = (common::GetEnv("MS_DEV_FALLBACK_USE_SUPPOSED_DTYPE") != "0");
  if (use_supposed_dtype) {
    auto any_abstract = cnode->abstract()->cast_ptr<abstract::AbstractAny>();
    MS_EXCEPTION_IF_NULL(any_abstract);
    if (any_abstract->supposed_tensor_dtype()) {
      return nullptr;
    }
  }

  if (!common::AnfAlgo::HasNodeAttr(kAttrNeedCast, cnode)) {
    return nullptr;
  }
  common::AnfAlgo::EraseNodeAttr(kAttrNeedCast, node);

  const auto default_type = abstract::AbstractAny::DefaultDtype()->type_id();
  auto cast_node = insert_cast_function_(fg, node, kOpFormat_DEFAULT, default_type, default_type, node->Shape());
  common::AnfAlgo::SetNodeAttr(kAttrAnyTypeCast, MakeValue(True), cast_node);
  if (fg->isa<session::KernelGraph>()) {
    auto kg = fg->cast_ptr<session::KernelGraph>();
    MS_EXCEPTION_IF_NULL(kg);
    kg->ReplaceInternalOutput(node, cast_node);
  }
  return cast_node;
}
}  // namespace opt
}  // namespace mindspore
