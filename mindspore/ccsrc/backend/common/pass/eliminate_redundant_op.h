/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_ELIMINATE_REDUNDANT_OP_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_ELIMINATE_REDUNDANT_OP_H_

#include <vector>
#include <string>
#include <utility>
#include "utils/hash_map.h"
#include "ir/anf.h"
#include "include/backend/optimizer/pattern_engine.h"
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
using ConditionFunc = std::function<bool(const CNodePtr &node1, const CNodePtr &node2)>;
using RedundantOpPair = std::pair<std::string, ConditionFunc>;
using KernelWithIndex = std::pair<CNodePtr, size_t>;

class BACKEND_EXPORT EliminateRedundantOp : public PatternProcessPass {
 public:
  explicit EliminateRedundantOp(bool multigraph = true) : PatternProcessPass("eliminate_redundant_op", multigraph) {
    Init();
  }
  ~EliminateRedundantOp() override = default;
  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const override;

 private:
  void Init();
  const AnfNodePtr DoEliminate(const FuncGraphPtr &func_graph, const CNodePtr &cnode) const;
  const AnfNodePtr ProcessMatchedNodes(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                       const CNodePtr &prev_cnode,
                                       std::vector<KernelWithIndex> *const pass_vector) const;
  mindspore::HashMap<std::string, RedundantOpPair> redundant_process_map_;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_ELIMINATE_REDUNDANT_OP_H_
