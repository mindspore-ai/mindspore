/**
 * Copyright 2021-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_LESS_BATCH_NORMALIZATION_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_LESS_BATCH_NORMALIZATION_H_

#include <utility>
#include <vector>
#include <tuple>
#include <string>

#include "frontend/optimizer/irpass.h"
#include "include/frontend/optimizer/optimizer.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"

namespace mindspore {
namespace opt {
namespace irpass {
using kStructureTuple = std::tuple<size_t, std::vector<PrimitivePtr>, std::pair<size_t, size_t>>;
class LessBatchNormalization : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) override;
  void Visit(const CNodePtr &cnode) override;
  void Reset();
  void IsRemoveNode(const CNodePtr &cnode, const std::vector<kStructureTuple> &match_pattern);
  bool MatchStructureNode(const CNodePtr &cnode, const int32_t index, const kStructureTuple &patternTuple) const;
  bool MatchGraphStructure(const CNodePtr &cnode, const std::vector<kStructureTuple> &match_pattern);

 private:
  mindspore::HashSet<CNodePtr> remove_node_list_{};
  std::vector<size_t> total_match_node_{0};
  size_t match_node_{0};
  size_t match_branch_{0};
  size_t match_pattern_{0};
  bool is_match_{false};
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_LESS_BATCH_NORMALIZATION_H_
