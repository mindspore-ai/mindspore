/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_DECREASE_TRANSPOSE_ALGO_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_DECREASE_TRANSPOSE_ALGO_H_

#include <vector>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include "include/backend/optimizer/optimizer.h"
#include "include/common/utils/utils.h"
#include "tools/optimizer/common/format_utils.h"
#include "tools/optimizer/format/delete_redundant_transpose.h"
#include "tools/optimizer/graph/transpose_strategy.h"

using mindspore::converter::FmkType;
namespace mindspore {
namespace opt {
class DecreaseTransposeAlgo : public Pass {
 public:
  explicit DecreaseTransposeAlgo(FmkType fmk_type = FmkType::kFmkTypeMs, bool train_flag = false,
                                 bool surrounded_all_trans = true)
      : Pass("DecreaseTransposeAlgo"), fmk_type_(fmk_type), train_flag_(train_flag), all_trans_(surrounded_all_trans) {}
  ~DecreaseTransposeAlgo() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  STATUS InsertPostTransNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode, const std::vector<int> &perm);
  STATUS GenNewInput(const FuncGraphPtr &func_graph, const CNodePtr &cnode, const std::vector<int> perm, bool before,
                     size_t index = 0);
  bool DecreaseTransposeForSingleOp(const FuncGraphPtr &func_graph);
  bool DecreaseTransposeForMultiOp(const FuncGraphPtr &func_graph);
  STATUS PostTransposeFusion(const FuncGraphPtr &func_graph, const CNodePtr &cnode);

  STATUS HandleGraphSingleNode(const FuncGraphPtr &func_graph, const TransTypePair &trans_info, const CNodePtr &cnode);
  STATUS HandleGraphMultiNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                              std::set<CNodePtr> *visit_transposes);
  STATUS InsertPreTransNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode, TransTypePair *trans_insert_info);
  STATUS DoPreInsert(const FuncGraphPtr &func_graph, const CNodePtr &cnode, FormatTransNodeType trans_type);
  STATUS InsertPreTransForNonTransInOut(const FuncGraphPtr &func_graph, const AnfNodeIndexSet &not_trans_in_nodes,
                                        const AnfNodeIndexSet &not_trans_out_nodes, TransTypePair trans_info);
  int SetSubGraphInput(const CNodePtr &cnode, const FuncGraphPtr &sub_graph);
  int ResetSubGraphInput();
  int SetSubGraphOutput(const FuncGraphPtr &sub_graph);
  int ModifyCNodeFormat(const CNodePtr &cnode, FormatTransNodeType pre_trans_type);
  bool IsNeedGenNewInput(const FuncGraphPtr &func_graph, const CNodePtr &cnode, size_t index);
  FmkType fmk_type_{converter::kFmkTypeMs};
  bool train_flag_{false};
  bool all_trans_{false};
  NodeInferShape node_infer_shape_;
  TransposeStrategy transpose_strategy_;
  DeleteRedundantTranspose delete_redundant_transpose_;
  std::unordered_map<FuncGraphPtr, std::vector<AnfNodePtr>> sub_inputs_map_;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_DECREASE_TRANSPOSE_ALGO_H_
