/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/ge/resize_bilinear_add_attr.h"

#include <memory>
#include "include/common/utils/anfalgo.h"
#include "ops/image_ops.h"
#include "ops/auto_generate/gen_ops_primitive.h"

namespace mindspore {
namespace opt {
namespace {
bool IsResizeBilinear(const BaseRef &ref) {
  if (utils::isa<AnfNodePtr>(ref)) {
    AnfNodePtr node = utils::cast<AnfNodePtr>(ref);
    MS_EXCEPTION_IF_NULL(node);
    if (IsPrimitive(node, prim::kPrimResizeBilinearV2)) {
      return true;
    }
  }

  return false;
}
}  // namespace

const BaseRef ResizeBilinearAddAttr::DefinePattern() const {
  VarPtr resize = std::make_shared<CondVar>(IsResizeBilinear);
  VarPtr inputs = std::make_shared<SeqVar>();
  return VectorRef({resize, inputs});
}

const AnfNodePtr ResizeBilinearAddAttr::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  CNodePtr cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  auto prim = common::AnfAlgo::GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(prim);

  prim->set_attr(kAttrDType, node->Type());

  return node;
}
}  // namespace opt
}  // namespace mindspore
