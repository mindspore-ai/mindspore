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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_PADV3_GE_PASS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_PADV3_GE_PASS_H_
#include <set>
#include <vector>
#include <string>
#include "include/backend/optimizer/pass.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore::opt {
class PadV3GePass : public Pass {
 public:
  PadV3GePass() : Pass("padv3_ge_pass") {}
  ~PadV3GePass() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  STATUS ProcessPadV3ForGE(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager);
  const CNodePtr CreateStridedSlice(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node, int64_t index);
  const CNodePtr ProcessSliceNConcat(const FuncGraphPtr &func_graph, const AnfNodePtr &pad_node,
                                     const AnfNodePtr &input_node, int64_t padding_length,
                                     std::string concat_node_name);
  const int64_t GetPaddingLength(const FuncGraphPtr &func_graph, const CNodePtr &pad_node);
  const ValueNodePtr GenerateDataValue(const FuncGraphPtr &func_graph, int64_t value);
  const ValueNodePtr GenerateDataValueTuple(const FuncGraphPtr &func_graph, int64_t value);
  const CNodePtr CreateConcatNode(const FuncGraphPtr &func_graph, const std::vector<AnfNodePtr> &concat_input_vec,
                                  std::string concat_node_name);
};
}  // namespace mindspore::opt
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_PADV3_GE_PASS_H_
