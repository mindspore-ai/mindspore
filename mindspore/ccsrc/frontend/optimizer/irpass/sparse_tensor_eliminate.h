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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_SPARSE_TENSOR_ELIMINATE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_SPARSE_TENSOR_ELIMINATE_H_

#include <vector>
#include <algorithm>

#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/optimizer.h"
#include "ir/visitor.h"

namespace mindspore {
namespace opt {
namespace irpass {
// {prim::kPrimSparseTensorGetIndices, {prim::kPrimMakeSparseTensor, Xs}}
// {prim::kPrimSparseTensorGetValues, {prim::kPrimMakeSparseTensor, Xs}}
// {prim::kPrimSparseTensorGetDenseShape, {prim::kPrimMakeSparseTensor, Xs}}
class SparseTensorEliminater : public OptimizerCaller {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    PatternNode x, y, z;
    auto sparse = PPrimitive(prim::kPrimMakeSparseTensor, x, y, z).MinExtraNodes(0);
    MATCH_REPLACE(node, PPrimitive(prim::kPrimSparseTensorGetIndices, sparse), x);
    MATCH_REPLACE(node, PPrimitive(prim::kPrimSparseTensorGetValues, sparse), y);
    MATCH_REPLACE(node, PPrimitive(prim::kPrimSparseTensorGetDenseShape, sparse), z);
    return nullptr;
  }
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_SPARSE_TENSOR_ELIMINATE_H_
