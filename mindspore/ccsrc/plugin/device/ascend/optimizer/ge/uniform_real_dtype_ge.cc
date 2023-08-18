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
#include "plugin/device/ascend/optimizer/ge/uniform_real_dtype_ge.h"
#include <memory>
#include "include/backend/optimizer/helper.h"
#include "include/common/utils/anfalgo.h"
#include "mindspore/core/ops/nn_ops.h"
#include "utils/anf_utils.h"

namespace mindspore {
namespace opt {
const BaseRef UniformRealDtypeGe::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimUniformReal, Xs});
}

const AnfNodePtr UniformRealDtypeGe::Process(const FuncGraphPtr &, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(node);
  auto uniform_real_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(uniform_real_cnode);
  auto prim = common::AnfAlgo::GetCNodePrimitive(uniform_real_cnode);
  MS_EXCEPTION_IF_NULL(prim);
  prim->set_attr(kAttrDType, MakeValue(kFloat32));
  return node;
}
}  // namespace opt
}  // namespace mindspore
