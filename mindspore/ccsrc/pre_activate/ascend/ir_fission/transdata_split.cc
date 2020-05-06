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
#include "pre_activate/ascend/ir_fission/transdata_split.h"
#include <set>
#include "pre_activate/ascend/ascend_helper.h"
#include "session/anf_runtime_algorithm.h"
#include "debug/anf_ir_dump.h"

namespace mindspore {
namespace opt {
const std::set<std::pair<string, string>> invalid_formats_pair = {{kOpFormat_C1HWNCoC0, kOpFormat_NCHW},
                                                                  {kOpFormat_NCHW, kOpFormat_C1HWNCoC0},
                                                                  {kOpFormat_C1HWNCoC0, kOpFormat_DEFAULT},
                                                                  {kOpFormat_DEFAULT, kOpFormat_C1HWNCoC0}};

bool TransDataSplit::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  bool changed = false;
  std::vector<AnfNodePtr> node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (node != nullptr && node->isa<CNode>() && AnfAlgo::GetCNodeName(node) == kTransDataOpName) {
      CheckCNodeInputSize(node->cast<CNodePtr>(), kBackendTransDataInputNum);
      if (IsFormatInvaild(node)) {
        changed = DoSplit(func_graph, node);
      }
    }
  }
  return changed;
}
bool TransDataSplit::IsFormatInvaild(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto input_format = AnfAlgo::GetInputFormat(node, 0);
  auto output_format = AnfAlgo::GetOutputFormat(node, 0);
  auto format_pair = std::make_pair(input_format, output_format);

  return invalid_formats_pair.find(format_pair) != invalid_formats_pair.end();
}
// transdata cannot support frac_z to nchw need split transdata(frac_z-HWCN) and transpose(HWCN-NCHW)
bool TransDataSplit::DoSplit(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto input_node = node->cast<CNodePtr>()->input(1);
  MS_EXCEPTION_IF_NULL(input_node);

  auto input_format = AnfAlgo::GetInputFormat(node, 0);
  auto output_format = AnfAlgo::GetOutputFormat(node, 0);
  AnfNodePtr new_transdata_node = nullptr;
  AnfNodePtr new_transpose_node = nullptr;
  AnfNodePtr new_replace_node = nullptr;
  // if output_format=default transdata need split transdata->transpose else transpose->transdata
  if (output_format == kOpFormat_DEFAULT || output_format == kOpFormat_NCHW) {
    // trans input_format to hwcn
    new_transdata_node =
      AddTransOpNodeToGraph(func_graph, node, kernel_select_, 0, input_format, kOpFormat_HWCN, kTransDataOpName, true);
    // trans hwcn to default_format
    new_transpose_node = AddTransOpNodeToGraph(func_graph, new_transdata_node, kernel_select_, 0, kOpFormat_HWCN,
                                               output_format, prim::kPrimTranspose->name(), false);
    AnfAlgo::SetNodeAttr(kAttrPerm, MakeValue(std::vector<int>{3, 2, 0, 1}), new_transpose_node);
    new_replace_node = new_transpose_node;
  } else {
    // trans default to hwcn
    new_transpose_node = AddTransOpNodeToGraph(func_graph, node, kernel_select_, 0, input_format, kOpFormat_HWCN,
                                               prim::kPrimTranspose->name(), true);
    AnfAlgo::SetNodeAttr(kAttrPerm, MakeValue(std::vector<int>{2, 3, 1, 0}), new_transpose_node);

    // trans hwcn to output_format
    new_transdata_node = AddTransOpNodeToGraph(func_graph, new_transpose_node, kernel_select_, 0, kOpFormat_HWCN,
                                               output_format, kTransDataOpName, false);
    new_replace_node = new_transdata_node;
  }
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(func_graph);

  if (!manager->Replace(node, new_replace_node)) {
    MS_LOG(EXCEPTION) << "Manager replace node failed";
  }
  MS_LOG(INFO) << "Transdata node:" << cnode->DebugString() << "split success.";
  return true;
}
}  // namespace opt
}  // namespace mindspore
