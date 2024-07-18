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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_ADAM_WEIGHT_DECAY_UNIFY_MINDIR_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_ADAM_WEIGHT_DECAY_UNIFY_MINDIR_H_

#include <vector>
#include <memory>
#include <string>
#include "include/backend/optimizer/optimizer.h"
#include "mindspore/core/ops/math_ops.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace opt {
constexpr size_t kAdamWeightDecaySingleOpInputNum = 11;

class AdamWeightDecayUnifyMindIR : public PatternProcessPass {
 public:
  explicit AdamWeightDecayUnifyMindIR(bool multigraph = true)
      : PatternProcessPass("adam_weight_decay_unify_mindir", multigraph) {}
  ~AdamWeightDecayUnifyMindIR() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  const AnfNodePtr CreateAdamApplyOneWithDecay(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                               const std::vector<AnfNodePtr> &ori_param,
                                               const AnfNodePtrList &input_list,
                                               const AnfNodePtrList &new_node_inputs) const;
  const AnfNodePtr CreateAdamApplyOneWithDecayAssign(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                     const AnfNodePtrList &input_list,
                                                     AnfNodePtrList *new_node_inputs) const;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_ADAM_WEIGHT_DECAY_UNIFY_MINDIR_H_
