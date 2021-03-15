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

#include "backend/optimizer/ascend/format_type/insert_transpose_for_dyanmic_gru_v2.h"
#include <memory>
#include <vector>
#include "utils/utils.h"
#include "backend/optimizer/ascend/ascend_helper.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "runtime/device/kernel_info.h"
#include "backend/kernel_compiler/oplib/oplib.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace opt {
const BaseRef InsertTransposeForDynamicGRUV2::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  VarPtr X1 = std::make_shared<Var>();
  VarPtr Xs = std::make_shared<SeqVar>();
  MS_EXCEPTION_IF_NULL(X);
  MS_EXCEPTION_IF_NULL(X1);
  MS_EXCEPTION_IF_NULL(Xs);
  return VectorRef(
    {prim::kPrimDynamicGRUV2, X1, VectorRef({prim::KPrimTransData, VectorRef({prim::kPrimReshape, X})}), Xs});
}

CNodePtr Insert(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(cnode);

  for (size_t index = 0; index < cnode->inputs().size(); index++) {
    if (index == 1 || index == 2) {
      AnfNodePtr new_node = nullptr;
      AnfNodePtr new_transdata_node = nullptr;
      AnfNodePtr new_transpose_node = nullptr;
      AnfNodePtr transdata_node = AnfAlgo::GetInputNode(cnode, index);
      AnfNodePtr reshape_node = AnfAlgo::GetInputNode(transdata_node->cast<CNodePtr>(), 0);
      auto input_format = AnfAlgo::GetInputFormat(transdata_node, 0);
      auto output_format = AnfAlgo::GetOutputFormat(transdata_node, 0);
      auto padding_axis = AnfAlgo::GetOutputReshapeType(transdata_node, 0);
      KernelSelectPtr kernel_select = std::make_shared<KernelSelect>();
      // trans default to hwcn
      new_transpose_node =
        NewTransOpNode(func_graph, AnfAlgo::GetInputNode(transdata_node->cast<CNodePtr>(), 0), kernel_select, false,
                       prim::kPrimTranspose->name(), std::vector<int64_t>{2, 3, 1, 0});
      AnfAlgo::SetNodeAttr("nop_op", MakeValue(true), new_transpose_node);
      RefreshKernelBuildInfo(input_format, kOpFormat_HWCN, new_transpose_node);
      // trans hwcn to output_format
      new_transdata_node =
        NewTransOpNode(func_graph, new_transpose_node, kernel_select, false, prim::KPrimTransData->name());
      RefreshKernelBuildInfo(kOpFormat_HWCN, output_format, new_transdata_node, padding_axis);
      new_transdata_node->set_abstract(transdata_node->abstract());
      new_node = new_transdata_node;

      FuncGraphManagerPtr manager = func_graph->manager();
      MS_EXCEPTION_IF_NULL(manager);
      manager->AddFuncGraph(func_graph);
      if (!manager->Replace(transdata_node, new_node)) {
        MS_LOG(EXCEPTION) << "For DynamicGRUV2, manager replace node failed";
      }
    }
  }
  return cnode;
}

const AnfNodePtr InsertTransposeForDynamicGRUV2::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                         const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto op_name = AnfAlgo::GetCNodeName(cnode);
  CNodePtr new_node = nullptr;
  if (op_name == kDynamicGRUV2OpName) {
    new_node = Insert(func_graph, cnode);
  }
  return new_node;
}
}  // namespace opt
}  // namespace mindspore
