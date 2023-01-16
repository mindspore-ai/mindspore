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
#include "plugin/device/ascend/optimizer/mindir/ascend_vm_op_adapter.h"

#include "include/common/utils/anfalgo.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "plugin/device/ascend/optimizer/create_node_helper.h"

namespace mindspore::opt {
const AnfNodePtr AscendVmOpAdapter::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  // There are other UnifyMindIR pass before AscendVmOpAdapter which may create new nodes.
  if (graph->has_flag(kAttrMutableKernel)) {
    AnfAlgo::SetDynamicAttrToPrim(common::AnfAlgo::GetCNodePrimitive(node));
  }
  return CreateNodeHelper::CreateNodeWithCheck(node);
}
}  // namespace mindspore::opt
