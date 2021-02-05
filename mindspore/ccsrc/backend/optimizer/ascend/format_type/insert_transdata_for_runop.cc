/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "backend/optimizer/ascend/format_type/insert_transdata_for_runop.h"
#include <memory>
#include "utils/utils.h"
#include "backend/optimizer/ascend/ascend_helper.h"
#include "backend/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
bool RunOpInsertTransData::Run(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  bool changed = false;
  std::vector<AnfNodePtr> node_list = TopoSort(graph->get_return());
  for (auto &node : node_list) {
    bool has_changed = false;
    MS_EXCEPTION_IF_NULL(node);
    if (!node->cast<CNodePtr>() || !AnfAlgo::IsRealKernel(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    size_t input_num = AnfAlgo::GetInputTensorNum(cnode);
    for (size_t index = 0; index < input_num; ++index) {
      auto prev_input_format = AnfAlgo::GetPrevNodeOutputFormat(cnode, index);
      auto prev_node_out_infer_shape = AnfAlgo::GetPrevNodeOutputInferShape(cnode, index);
      auto input_format = AnfAlgo::GetInputFormat(cnode, index);
      auto input_node = AnfAlgo::GetInputNode(cnode, index);
      // convert the format of node's input node to default
      if (kCommonFormatSet.find(prev_input_format) == kCommonFormatSet.end() && prev_node_out_infer_shape.size() > 1) {
        auto trans_node = AddTransOpNodeToGraph(graph, input_node, kernel_select_, 0, false);
        AnfAlgo::SetNodeInput(cnode, trans_node, index);
        has_changed = true;
      }
      // convert node's output format
      if (kCommonFormatSet.find(input_format) == kCommonFormatSet.end() && prev_node_out_infer_shape.size() > 1) {
        auto trans_node = AddTransOpNodeToGraph(graph, cnode, kernel_select_, index, true);
        AnfAlgo::SetNodeInput(cnode, trans_node, index);
        has_changed = true;
      }
    }
    if (has_changed) {
      auto kernel_graph = graph->cast<KernelGraphPtr>();
      MS_EXCEPTION_IF_NULL(kernel_graph);
      auto new_node = kernel_graph->NewCNode(cnode);
      auto manager = kernel_graph->manager();
      MS_EXCEPTION_IF_NULL(manager);
      manager->Replace(cnode, new_node);
      changed = true;
    }
  }
  return changed;
}
}  // namespace opt
}  // namespace mindspore
