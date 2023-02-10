/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_ALL_TO_ALL_UNIFY_MINDIR_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_ALL_TO_ALL_UNIFY_MINDIR_H_

#include <memory>
#include <string>
#include <vector>
#include "backend/common/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class NeighborExchangeUnifyMindIR : public PatternProcessPass {
 public:
  explicit NeighborExchangeUnifyMindIR(bool multigraph = true)
      : PatternProcessPass("neighbor_exchange_unify_mindir", multigraph) {}
  ~NeighborExchangeUnifyMindIR() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  std::vector<std::string> MustExistPrimitiveName() const override;
};

class AllToAllUnifyMindIR : public PatternProcessPass {
 public:
  explicit AllToAllUnifyMindIR(bool multigraph = true) : PatternProcessPass("all_to_all_unify_mindir", multigraph) {}
  ~AllToAllUnifyMindIR() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  CNodePtr CreateSplitNode(const FuncGraphPtr &graph, const CNodePtr &all_to_all) const;
  CNodePtr CreateAllToAllvNode(const FuncGraphPtr &graph, const CNodePtr &all_to_all, const CNodePtr &split) const;
  CNodePtr CreateConcatNode(const FuncGraphPtr &graph, const CNodePtr &all_to_all, const CNodePtr &all_to_all_v) const;
  std::vector<std::string> MustExistPrimitiveName() const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_ALL_TO_ALL_UNIFY_MINDIR_H_
