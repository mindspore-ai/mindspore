/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_SCALAR_OP_PASS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_SCALAR_OP_PASS_H_
#include <set>
#include "include/registry/converter_context.h"
#include "include/backend/optimizer/pass.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore::opt {
class ScalarOpPass : public Pass {
 public:
  ScalarOpPass() : Pass("scalar_op_pass") {}
  ~ScalarOpPass() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  ValueNodePtr GenerateScalarValueTensor(const FuncGraphPtr &func_graph, const AnfNodePtr &anf_node, int input_index);
  ValueNodePtr GenerateScalarValueTuple(const FuncGraphPtr &func_graph, int64_t value);
  CNodePtr GenerateScalarToTensor(const FuncGraphPtr &func_graph, const AnfNodePtr &anf_node, int input_index);
  CNodePtr GenerateTensorToScalar(const FuncGraphPtr &func_graph, const AnfNodePtr &anf_node,
                                  bool is_curr_node = false);
  CNodePtr GenerateTensorShape(const FuncGraphPtr &func_graph, const AnfNodePtr &anf_node);
  CNodePtr GenerateStridedSlice(const FuncGraphPtr &func_graph, const AnfNodePtr &shape_node,
                                const AnfNodePtr &tuple_get_node, const FuncGraphManagerPtr &manager);
  STATUS ReplaceScalarOp(const FuncGraphPtr &func_graph, const AnfNodePtr &anf_node, const FuncGraphManagerPtr &manager,
                         const PrimitivePtr &replace_op_prim);
  STATUS ReplaceMakeTuple(const FuncGraphPtr &func_graph, const AnfNodePtr &anf_node,
                          const FuncGraphManagerPtr &manager);
  STATUS ReplaceShapeTupleGet(const FuncGraphPtr &func_graph, const AnfNodePtr &anf_node,
                              const FuncGraphManagerPtr &manager);
  STATUS RemoveTensorToScalar(const FuncGraphPtr &func_graph, const AnfNodePtr &anf_node,
                              const FuncGraphManagerPtr &manager);
  size_t GetInputNodeIndex(const AnfNodePtr &input, const CNodePtr &user_node);
  STATUS RunScalarOpPass(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager);
  STATUS RunMakeTuplePass(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager);
  STATUS RunShapeTupleGetPass(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager);
  STATUS RunRemoveTensorToScalarPass(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager);
  STATUS RunArithmeticCheckPass(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager);
};
}  // namespace mindspore::opt
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_SCALAR_OP_PASS_H_
