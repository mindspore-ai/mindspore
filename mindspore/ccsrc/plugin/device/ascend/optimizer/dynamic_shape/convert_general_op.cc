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

#include "plugin/device/ascend/optimizer/dynamic_shape/convert_general_op.h"

#include <memory>
#include "backend/common/session/anf_runtime_algorithm.h"
#include "backend/common/optimizer/helper.h"
#include "plugin/device/ascend/optimizer/dynamic_shape/ascend_dynamic_shape_helper.h"

namespace mindspore {
namespace opt::dynamic_shape {
const BaseRef ConvertGeneralOp::DefinePattern() const {
  VarPtr X = std::make_shared<CondVar>(IsGeneralOp);
  return BaseRef({X});
}

const AnfNodePtr ConvertGeneralOp::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto stub_infer_node = GenInferNode(node, true);
  auto init_node = GenInitNode(node);
  auto sync_node = GenUpdateNode(node, true);  // Use to call sync if needed.
  RelatedCustomActorNode custom_nodes = {stub_infer_node, init_node, sync_node};
  CustomActorNodeManager::Instance().Register(node, custom_nodes);
  return node;
}
}  // namespace opt::dynamic_shape
}  // namespace mindspore
