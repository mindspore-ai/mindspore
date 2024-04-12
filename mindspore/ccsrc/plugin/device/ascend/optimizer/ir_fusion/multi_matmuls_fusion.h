/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_MULTI_MATMULS_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_MULTI_MATMULS_FUSION_H_

#include <string>
#include <memory>

#include "include/backend/optimizer/pass.h"
#include "ir/func_graph.h"
#include "ir/anf.h"
#include "include/backend/optimizer/helper.h"
#include "include/backend/optimizer/optimizer.h"
#include "mindspore/core/ops/nn_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/sequence_ops.h"

namespace mindspore {
namespace opt {
/**
 * Fuse MatMul when a node is used by several matmuls.
 *
 * example:
 * x = MatMul(A, B, false, false)
 * y = MatMul(A, C, false, true)
 * z = MatMul(A, D, false, false)
 * ...
 * ------->
 * t = MatmulQkv(A, B, false, false, C, false, true, D, false, false) # or MatmulFfn
 * x = tuple_getitem(t, 0)
 * y = tuple_getitem(t, 1)
 * z = tuple_getitem(t, 2)
 * ...
 */
class MultiMatmulsFusion : public Pass {
 public:
  MultiMatmulsFusion() : Pass("multi_matmuls_fusion") {}
  ~MultiMatmulsFusion() override = default;
  bool Run(const FuncGraphPtr &graph) override;

 protected:
  void Process(const std::string &name, const AnfNodePtr &node, const AnfNodePtrList &users,
               AnfNodePtrList *getitems) const;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_MULTI_MATMULS_FUSION_H_
