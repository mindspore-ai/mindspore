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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_GROUPED_MATMUL_OP_PASS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_GROUPED_MATMUL_OP_PASS_H_
#include <vector>
#include <string>
#include <set>
#include "include/registry/converter_context.h"
#include "include/backend/optimizer/pass.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore::opt {
class GroupedMatmulOpPass : public Pass {
 public:
  GroupedMatmulOpPass() : Pass("grouped_matmul_op_pass") {}
  ~GroupedMatmulOpPass() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  STATUS RunInsertSizeAttrPass(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager);
  bool IsTupleHasDynamicSequence(const abstract::AbstractBasePtr &abstract);
  size_t GetOutputElementNum(const AnfNodePtr &node);
  CNodePtr NewCNode(const std::vector<AnfNodePtr> &inputs, const FuncGraphPtr &fg,
                    const std::vector<AnfNodePtr> &orig_nodes);
  CNodePtr CreateTupleGetItemNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node, size_t output_idx);
  void UseEmptyNodeReplaceNone(const FuncGraphPtr &graph, const std::string &cnode_name, const size_t input_idx,
                               std::vector<int64_t> *dyn_input_sizes, std::vector<AnfNodePtr> *plant_inputs);
  int64_t SplitTupleInputs(const FuncGraphPtr &graph, const AnfNodePtr &tuple_input,
                           std::vector<AnfNodePtr> *plant_inputs);
  bool IsNotSequenceOfTensor(const abstract::AbstractBasePtr &abs);
  AnfNodePtr ConvertMakeTupleInputToPlantInputs(const FuncGraphPtr &graph, const CNodePtr &cnode_ptr);
};
}  // namespace mindspore::opt
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_GROUPED_MATMUL_OP_PASS_H_
