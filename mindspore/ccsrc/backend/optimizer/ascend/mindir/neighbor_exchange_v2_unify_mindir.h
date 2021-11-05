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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_NEIGHBOR_EXCHANGE_V2_UNIFY_MINDIR_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_NEIGHBOR_EXCHANGE_V2_UNIFY_MINDIR_H_

#include <memory>
#include <vector>
#include "backend/optimizer/common/optimizer.h"
#include "backend/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
class NeighborExchangeV2UnifyMindIR : public PatternProcessPass {
 public:
  explicit NeighborExchangeV2UnifyMindIR(bool multigraph = true)
      : PatternProcessPass("neighbor_exchange_v2_unify_mindir", multigraph) {}
  ~NeighborExchangeV2UnifyMindIR() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;
};

class NeighborExchangeV2GradUnifyMindIR : public PatternProcessPass {
 public:
  explicit NeighborExchangeV2GradUnifyMindIR(bool multigraph = true)
      : PatternProcessPass("neighbor_exchange_v2_grad_unify_mindir", multigraph) {}
  ~NeighborExchangeV2GradUnifyMindIR() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;
};

}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_NEIGHBOR_EXCHANGE_V2_UNIFY_MINDIR_H_
