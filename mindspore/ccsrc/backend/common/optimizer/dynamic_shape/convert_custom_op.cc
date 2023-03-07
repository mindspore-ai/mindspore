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

#include "backend/common/optimizer/dynamic_shape/convert_custom_op.h"

#include <memory>
#include <string>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/optimizer/helper.h"
#include "backend/common/optimizer/dynamic_shape/dynamic_shape_helper.h"
#include "utils/ms_context.h"
#include "utils/anf_utils.h"

namespace mindspore {
namespace opt::dynamic_shape {
bool ConvertCustomOp::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto node_list = TopoSort(func_graph->get_return());
  for (const auto &node : node_list) {
    if (!IsRealCNode(node)) {
      continue;
    }
    ConvertCustomOpForNode(node);
  }
  return true;
}

void ConvertCustomOp::ConvertCustomOpForNode(const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(node);
  bool is_dynamic_node = common::AnfAlgo::IsDynamicShape(node);
  AnfNodePtr infer_node = nullptr;
  AnfNodePtr init_node = nullptr;
  if (is_dynamic_node) {
    infer_node = GenInferNode(node);
    init_node = GenInitNode(node);
    AnfUtils::SetCustomInfoToBaseNode(node, infer_node, init_node);
  }

  RelatedCustomActorNode custom_nodes = {infer_node, init_node};
  CustomActorNodeManager::Instance().Register(node, custom_nodes);
}
}  // namespace opt::dynamic_shape
}  // namespace mindspore
