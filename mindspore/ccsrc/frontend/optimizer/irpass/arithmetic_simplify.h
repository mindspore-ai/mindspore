/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_ARITHMETIC_SIMPLIFY_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_ARITHMETIC_SIMPLIFY_H_

#include <memory>

#include "frontend/optimizer/irpass.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/other_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "frontend/optimizer/irpass/prim_eliminate.h"
#include "frontend/optimizer/optimizer_caller.h"
#include "frontend/optimizer/anf_visitor.h"
#include "ir/pattern_matcher.h"

namespace mindspore {
namespace opt {
namespace irpass {
// grad = AllReduce(grad) / worker_number
// grad = grad + weight * decy
// ->
// grad = grad + weight * decy
// grad = AllReduce(grad) / worker_number

// {prim::kPrimAddN, {prim::kPrimMakeTuple, {prim::kPrimMul, {prim::kPrimAllReduce, X}, Y}, Z}} ->
// {prim::kPrimMul, {prim::kPrimAllReduce, {prim::kPrimAddN,{prim::kPrimMakeTuple, Z, X}}}, Y}
class AdjustAllReduceMulAdd : public OptimizerCaller {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;

  void ProcessDependEdge(const FuncGraphPtr &fg, const AnfNodePtr &addn_maketuple, const AnfNodePtr &new_node);

 private:
  AnfNodePtr mul_cnode_{nullptr};
};

class ArithmeticSimplify : public OptimizerCaller {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_ARITHMETIC_SIMPLIFY_H_
