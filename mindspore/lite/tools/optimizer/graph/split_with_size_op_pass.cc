/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include <memory>
#include <vector>
#include <string>
#include "ops/auto_generate/gen_lite_ops.h"
#include "ops/auto_generate/gen_ops_primitive.h"
#include "ops/array_ops.h"
#include "ops/lite_ops.h"
#include "ops/tuple_get_item.h"
#include "ops/make_tuple.h"
#include "tools/optimizer/graph/split_with_size_op_pass.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/optimizer/graph/lite_tensor_extractor.h"
#include "mindspore/core/abstract/ops/primitive_infer_map.h"
#include "mindspore/core/utils/anf_utils.h"
#include "mindspore/core/ops/math_ops.h"
#include "extendrt/utils/func_graph_utils.h"
#include "include/common/utils/anfalgo.h"
#include "tools/converter/quantizer/quantize_util.h"

namespace mindspore::opt {

AnfNodePtr SplitWithSizeOpPass::SplitWithSizeMapperToSplitV(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(func_graph);

  auto input_x_node = cnode->input(kInputIndexOne);
  MS_EXCEPTION_IF_NULL(input_x_node);
  auto size_splits_node = cnode->input(kInputIndexTwo);
  MS_EXCEPTION_IF_NULL(size_splits_node);
  auto split_dim_node = cnode->input(kInputIndexThree);
  MS_EXCEPTION_IF_NULL(split_dim_node);

  auto size_splits_value_node = size_splits_node->cast<ValueNodePtr>();
  auto size_splits_tensor = size_splits_value_node->value();
  auto size_splits_tensor_shape = GetValue<std::vector<int64_t>>(size_splits_tensor);
  auto num_split = size_splits_tensor_shape.size();

  auto split_dim = split_dim_node->cast<ValueNodePtr>();
  auto split_dim_tensor = split_dim->value();

  auto splitv_prim = prim::kPrimSplitV->Clone();
  MS_CHECK_TRUE_RET(splitv_prim, {});
  splitv_prim->AddAttr(kAttrNameNumSplit, MakeValue<int64_t>(num_split));
  splitv_prim->AddAttr(kAttrNameSizeSplits, size_splits_tensor);
  splitv_prim->AddAttr(kAttrNameSplitDim, split_dim_tensor);

  std::vector<AnfNodePtr> splitv_inputs = {input_x_node};

  auto splitv_cnode = func_graph->NewCNode(splitv_prim, splitv_inputs);
  splitv_cnode->set_abstract(cnode->abstract()->Clone());
  splitv_cnode->set_fullname_with_scope(cnode->fullname_with_scope() + "_to_splitv");
  MS_LOG(INFO) << "Create SplitV Node Success.";
  return splitv_cnode;
}

STATUS SplitWithSizeOpPass::RunSplitWithSizePass(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager) {
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    if (CheckPrimitiveType(node, prim::kPrimSplitWithSize)) {
      MS_LOG(INFO) << "Run SplitWithSize Pass for SplitWithSize node " << node->fullname_with_scope();
      auto new_cnode = this->SplitWithSizeMapperToSplitV(func_graph, node->cast<CNodePtr>());
      if (!new_cnode) {
        MS_LOG(ERROR) << "Fail to SplitWithSizeMapperToSplitV";
        return lite::RET_ERROR;
      } else {
        MS_LOG(INFO) << "SplitWithSize op pass create new node: " << new_cnode->fullname_with_scope();
        if (!manager->Replace(node, new_cnode)) {
          MS_LOG(ERROR) << "SplitWithSize op pass replace node " << node->fullname_with_scope() << " failed";
          return lite::RET_ERROR;
        }
      }
    }
  }
  return lite::RET_OK;
}

bool SplitWithSizeOpPass::Run(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_RET(manager != nullptr, false);
  auto status = RunSplitWithSizePass(func_graph, manager);
  MS_CHECK_TRUE_RET(status != lite::RET_ERROR, false);
  return true;
}
}  // namespace mindspore::opt
