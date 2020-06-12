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

#include "pre_activate/ascend/ir_fusion/remove_reshape_pair.h"
#include <memory>
#include "session/anf_runtime_algorithm.h"
#include "utils/utils.h"
#include "operator/ops.h"

namespace mindspore {
namespace opt {
const BaseRef RemoveReshapePair::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  MS_EXCEPTION_IF_NULL(X);
  return VectorRef({prim::kPrimReshape, VectorRef({prim::kPrimReshape, X})});
}

const AnfNodePtr RemoveReshapePair::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                            const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(equiv);
  auto reshape_op_1 = CheckAnfNodeIfCNodeAndInputSize(node, kBackendReshapeInputNum);
  MS_EXCEPTION_IF_NULL(reshape_op_1);
  // If reshape operator used by more than one other operators, reshape operator cant not be deleted  directly
  if (IsUsedByOthers(func_graph, reshape_op_1)) {
    return nullptr;
  }
  auto reshape_op_2 = CheckAnfNodeIfCNodeAndInputSize(reshape_op_1->input(1), kBackendReshapeInputNum);
  MS_EXCEPTION_IF_NULL(reshape_op_2);
  if (IsUsedByOthers(func_graph, reshape_op_2)) {
    return nullptr;
  }
  auto output_shape = AnfAlgo::GetOutputDeviceShape(reshape_op_2, 0);
  auto input_shape = AnfAlgo::GetInputDeviceShape(reshape_op_1, 0);
  if (input_shape == output_shape) {
    auto input_node = reshape_op_2->input(1);
    return input_node;
  }
  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
