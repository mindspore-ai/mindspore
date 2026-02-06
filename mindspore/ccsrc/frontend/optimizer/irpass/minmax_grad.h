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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_MINMAX_GRAD_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_MINMAX_GRAD_H_

#include <vector>
#include <memory>

#include "include/frontend/optimizer/optimizer.h"
#include "primitive/sequence_ops.h"
#include "primitive/math_ops.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"

namespace mindspore {
namespace opt {
namespace irpass {
// {prim::kPrimTupleGetItem, {target_grad, Xs}, C}
class MinMaximumGrad : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) override;

  void Visit(const CNodePtr &cnode) override;

  void Visit(const ValueNodePtr &vnode) override;

  void Reset();

  // Check if node is MinimumGrad() or MaximumGrad()
  static bool IsOriginMaxMinGrad(const AnfNodePtr &node);

 private:
  int64_t idx_{-1};
  CNodePtr grad_{nullptr};
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_MINMAX_GRAD_H_
