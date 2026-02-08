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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_CDIST_FISSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_CDIST_FISSION_H_

#include <vector>
#include <string>
#include "include/backend/common/pass_manager/optimizer.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace opt {
class BACKEND_COMMON_EXPORT CdistFission : public PatternProcessPass {
 public:
  explicit CdistFission(bool multigraph = true) : PatternProcessPass("cdist_fission", multigraph) {}
  ~CdistFission() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const override;

 private:
  std::vector<std::string> MustExistPrimitiveName() const override;
};

class BACKEND_COMMON_EXPORT CdistGradFission : public PatternProcessPass {
 public:
  explicit CdistGradFission(bool multigraph = true) : PatternProcessPass("cdist_grad_fission", multigraph) {}
  ~CdistGradFission() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const override;

 private:
  std::vector<std::string> MustExistPrimitiveName() const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_CDIST_FISSION_H_
