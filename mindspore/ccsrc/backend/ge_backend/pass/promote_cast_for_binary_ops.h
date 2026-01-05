/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_GE_BACKEND_PASS_PROMOTE_CAST_FOR_BINARY_OPS_H_
#define MINDSPORE_CCSRC_BACKEND_GE_BACKEND_PASS_PROMOTE_CAST_FOR_BINARY_OPS_H_

#include "include/backend/common/pass_manager/optimizer.h"

namespace mindspore {
namespace opt {
class PromoteCastForBinaryOp : public PatternProcessPass {
 public:
  explicit PromoteCastForBinaryOp(const PrimitivePtr &prim, bool multi_graph = true)
      : PatternProcessPass("promote_cast_for_" + prim->name(), multi_graph), prim_(prim) {}
  ~PromoteCastForBinaryOp() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const override;

 private:
  PrimitivePtr prim_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_GE_BACKEND_PASS_PROMOTE_CAST_FOR_BINARY_OPS_H_
