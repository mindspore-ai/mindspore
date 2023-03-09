/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_MAXPOOL_TO_MAXPOOL_WITH_ARGMAX_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_MAXPOOL_TO_MAXPOOL_WITH_ARGMAX_H_

#include <vector>
#include <memory>
#include <string>
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class MaxPool2MaxPoolWithArgmax : public PatternProcessPass {
 public:
  explicit MaxPool2MaxPoolWithArgmax(bool multigraph = true)
      : PatternProcessPass("maxpool_to_maxpool_with_argmax", multigraph) {}
  ~MaxPool2MaxPoolWithArgmax() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  CNodePtr CreateMaxPoolWithArgmax(const FuncGraphPtr &graph, const CNodePtr &maxpool) const;
  CNodePtr CreateMaxPoolGradWithArgmax(const FuncGraphPtr &graph, const CNodePtr &maxpool_grad,
                                       const std::vector<AnfNodePtr> &maxpool_argmax_outputs) const;
  void SetNodeAttrs(const CNodePtr &maxpool, const CNodePtr &maxpool_grad, const CNodePtr &maxpool_argmax,
                    const CNodePtr &maxpool_grad_argmax) const;
  std::vector<std::string> MustExistPrimitiveName() const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_MAXPOOL_TO_MAXPOOL_WITH_ARGMAX_H_
