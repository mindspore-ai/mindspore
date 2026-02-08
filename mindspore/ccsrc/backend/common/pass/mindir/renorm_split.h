/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_RENORM_SPLIT_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_RENORM_SPLIT_H_
#include <vector>
#include <string>
#include <memory>
#include "ir/func_graph.h"
#include "include/backend/common/pass_manager/optimizer.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace opt {
class BACKEND_COMMON_EXPORT RenormSplit : public PatternProcessPass {
 public:
  explicit RenormSplit(bool multigraph = true, const std::string &name = "renorm_split")
      : PatternProcessPass(name, multigraph) {}
  ~RenormSplit() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const override;

 private:
  std::vector<std::string> MustExistPrimitiveName() const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_RENORM_SPLIT_H_
