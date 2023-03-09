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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_INPLACE_ASSIGN_FOR_CUSTOM_OP_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_INPLACE_ASSIGN_FOR_CUSTOM_OP_H_

#include <string>
#include <vector>

#include "ir/anf.h"
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class InplaceAssignForCustomOp : public PatternProcessPass {
 public:
  explicit InplaceAssignForCustomOp(bool multigraph = true)
      : PatternProcessPass("inplace_assign_for_custom_op", multigraph) {}
  ~InplaceAssignForCustomOp() override = default;
  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const override;
  std::vector<std::string> MustExistPrimitiveName() const override;

 private:
  mutable mindspore::HashSet<CNodePtr> visited_{};
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_INPLACE_ASSIGN_FOR_CUSTOM_OP_H_
