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

#include "plugin/device/ascend/optimizer/format_type/insert_transpose_for_dyanmic_gru_v2.h"
#include <memory>
#include <vector>
#include "include/common/utils/utils.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/kernel_info.h"
#include "kernel/oplib/oplib.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace opt {
constexpr size_t kInputIndex1 = 1;
constexpr size_t kInputIndex2 = 2;

const BaseRef InsertTransposeForDynamicGRUV2::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  VarPtr X1 = std::make_shared<Var>();
  VarPtr Xs = std::make_shared<SeqVar>();
  MS_EXCEPTION_IF_NULL(X);
  MS_EXCEPTION_IF_NULL(X1);
  MS_EXCEPTION_IF_NULL(Xs);
  return VectorRef(
    {prim::kPrimDynamicGRUV2, X1, VectorRef({prim::kPrimTransData, VectorRef({prim::kPrimReshape, X})}), Xs});
}

CNodePtr Insert(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(cnode);

  for (size_t index = 0; index < cnode->inputs().size(); index++) {
    if (index == kInputIndex1 || index == kInputIndex2) {
      AnfNodePtr new_node = nullptr;
      AnfNodePtr new_transdata_node = nullptr;
      AnfNodePtr new_transpose_node = nullptr;
      AnfNodePtr transdata_node = common::AnfAlgo::GetInputNode(cnode, index);
      auto input_format = AnfAlgo::GetInputFormat(transdata_node, 0);
      auto output_format = AnfAlgo::GetOutputFormat(transdata_node, 0);
      auto padding_axis = AnfAlgo::GetOutputReshapeType(transdata_node, 0);
      KernelSelectPtr kernel_select = std::make_shared<KernelSelect>();
      // trans default to hwcn
      new_transpose_node =
        NewTransOpNode(func_graph, common::AnfAlgo::GetInputNode(transdata_node->cast<CNodePtr>(), 0), cnode,
                       kernel_select, false, prim::kPrimTranspose->name(), std::vector<int64_t>{2, 3, 1, 0});
      MS_EXCEPTION_IF_NULL(new_transpose_node);
      // This Transpose operator is only to change the shape, but does not expect to change the data arrangement!
      common::AnfAlgo::SetNodeAttr(kAttrNopOp, MakeValue(true), new_transpose_node);
      RefreshKernelBuildInfo(kernel_select, input_format, kOpFormat_HWCN, new_transpose_node);
      // trans hwcn to output_format
      new_transdata_node =
        NewTransOpNode(func_graph, new_transpose_node, cnode, kernel_select, false, prim::kPrimTransData->name());
      MS_EXCEPTION_IF_NULL(new_transdata_node);
      RefreshKernelBuildInfo(kernel_select, kOpFormat_HWCN, output_format, new_transdata_node, padding_axis);
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
  common::AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto op_name = common::AnfAlgo::GetCNodeName(cnode);
  CNodePtr new_node = nullptr;
  if (op_name == kDynamicGRUV2OpName) {
    new_node = Insert(func_graph, cnode);
  }
  return new_node;
}
}  // namespace opt
}  // namespace mindspore
