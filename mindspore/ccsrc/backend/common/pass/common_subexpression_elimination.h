/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_COMMON_SUBEXPRESSION_ELIMINATION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_COMMON_SUBEXPRESSION_ELIMINATION_H_
#include <string>
#include "include/backend/visible.h"
#include "backend/common/optimizer/pass.h"
#include "include/common/utils/cse.h"

namespace mindspore {
namespace opt {
class BACKEND_EXPORT CommonSubexpressionElimination : public Pass {
 public:
  explicit CommonSubexpressionElimination(const std::string &name = "cse") : Pass(name) {}
  ~CommonSubexpressionElimination() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;
};

class BackendCSE : public CSE {
 public:
  BackendCSE() = default;
  ~BackendCSE() override = default;
  virtual bool CheckEqualCnodeInputs(const AnfNodePtr &main, const AnfNodePtr &node) const;
  bool CheckReplace(const AnfNodePtr &main, const AnfNodePtr &node) override;
  virtual bool CheckEqualKernelBuildInfo(const AnfNodePtr &main, const AnfNodePtr &node) const;
  bool Cse(const FuncGraphPtr graph, const FuncGraphManagerPtr manager) override;

 private:
  bool CheckValueNode(const ValueNodePtr &main, const ValueNodePtr &node) const;
  bool CheckCNode(const CNodePtr &main, const CNodePtr &node);
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_COMMON_SUBEXPRESSION_ELIMINATION_H_
