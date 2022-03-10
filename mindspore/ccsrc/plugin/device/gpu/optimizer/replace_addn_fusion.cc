/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "plugin/device/gpu/optimizer/replace_addn_fusion.h"
#include <vector>
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"
#include "backend/common/optimizer/helper.h"

namespace mindspore {
namespace opt {
const BaseRef ReplaceAddNFusion::DefinePattern() const {
  VectorRef addn = VectorRef({prim::kPrimAddN, A, B});
  return addn;
}

const AnfNodePtr ReplaceAddNFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto A = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), 0);
  auto B = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), 1);
  MS_EXCEPTION_IF_NULL(A);
  MS_EXCEPTION_IF_NULL(B);
  int64_t num_input = common::AnfAlgo::GetNodeAttr<int64_t>(node, "n");
  if (num_input == kAddNInputNum) {
    auto prim = std::make_shared<Primitive>(prim::kPrimAdd->name());
    MS_EXCEPTION_IF_NULL(prim);
    std::vector<AnfNodePtr> inputs = {NewValueNode(prim), A, B};
    auto add_new = graph->NewCNode(inputs);
    MS_EXCEPTION_IF_NULL(add_new);
    std::vector<TypeId> outputs_type;
    std::vector<BaseShapePtr> outputs_shape;
    outputs_type.push_back(common::AnfAlgo::GetOutputInferDataType(A, 0));
    outputs_shape.push_back(common::AnfAlgo::GetOutputDetailShape(A, 0));
    common::AnfAlgo::SetOutputTypeAndDetailShape(outputs_type, outputs_shape, add_new.get());
    auto manager = graph->manager();
    MS_EXCEPTION_IF_NULL(manager);
    manager->Replace(utils::cast<CNodePtr>(node), utils::cast<CNodePtr>(add_new));
    return add_new;
  } else {
    return nullptr;
  }
}
}  // namespace opt
}  // namespace mindspore
