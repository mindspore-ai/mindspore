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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_RESHAPE_ELIMINATE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_RESHAPE_ELIMINATE_H_

#include <vector>

#include "ir/func_graph.h"
#include "primitive/array_ops.h"
#include "frontend/optimizer/optimizer_caller.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"
#include "include/frontend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
namespace irpass {
using abstract::Shape;
using abstract::ShapePtr;

// {reshape_op, X, Shape}
class ReshapeSameShapeEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;
  void Visit(const AnfNodePtr &node) override;

  void Reset();

 private:
  AnfNodePtr x_{nullptr}, shape_{nullptr};
};

// {PrimReshape, {PrimReshape, X, Y}, Shape}
class TwoReshapeEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;

  void Visit(const AnfNodePtr &node) override;

  void Reset();

 private:
  PrimitivePtr prim_{nullptr};
  AnfNodePtr x_{nullptr}, shape_{nullptr};
};

class ReshapeEliminater : public OptimizerCaller {
 public:
  ReshapeEliminater();
  ~ReshapeEliminater() = default;

  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) override;

 private:
  ReshapeSameShapeEliminater reshape_same_shape_eliminater_;
  TwoReshapeEliminater two_reshape_eliminater_;
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_RESHAPE_ELIMINATE_H_
