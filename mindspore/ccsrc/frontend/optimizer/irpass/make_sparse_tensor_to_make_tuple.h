/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_MAKE_SPARSE_TENSOR_TO_MAKE_TUPLE_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_MAKE_SPARSE_TENSOR_TO_MAKE_TUPLE_

#include <vector>

#include "frontend/operator/ops.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
namespace irpass {
// {prim::kPrimMakeRowTensor, x, y, z} -> {prim::kPrimMakeTuple, x, y, z}
// {prim::kPrimMakeCOOTensor, x, y, z} -> {prim::kPrimMakeTuple, x, y, z}
// {prim::kPrimMakeCSRTensor, x, y, z, k} -> {prim::kPrimMakeTuple, x, y, z, k}
class MakeSparseTensorToMakeTuple : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    PatternNode x, y, z, k;
    auto make_row_tensor = PPrimitive(prim::kPrimMakeRowTensor, x, y, z).MinExtraNodes(0);
    auto make_coo_tensor = PPrimitive(prim::kPrimMakeCOOTensor, x, y, z).MinExtraNodes(0);
    auto make_csr_tensor = PPrimitive(prim::kPrimMakeCSRTensor, x, y, z, k).MinExtraNodes(0);
    MATCH_REPLACE(node, make_row_tensor, PPrimitive(prim::kPrimMakeTuple, x, y, z));
    MATCH_REPLACE(node, make_coo_tensor, PPrimitive(prim::kPrimMakeTuple, x, y, z));
    MATCH_REPLACE(node, make_csr_tensor, PPrimitive(prim::kPrimMakeTuple, x, y, z, k));
    return nullptr;
  }
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_ROW_TENSOR_ELIMINATE_H_
