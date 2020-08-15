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

#include <algorithm>
#include <memory>
#include <vector>

#include "frontend/optimizer/irpass.h"
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

// Arithmetic Simplifications should be done after step_parallel.
// eg: Mul(0, weight) where weight is a parameter will be simplified to a constant tensor
// with shape(weight), but after step_parallel, shape of weight may be changed, so the
// shape of the constant tensor should also be changed. So this pass is seperated from
// ArithmeticSimplify and deferred until step_parallel.
class ArithmeticSimplify2 : public OptimizerCaller {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_ARITHMETIC_SIMPLIFY_H_
