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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_MULTI_WEIGHT_MATMULS_2_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_MULTI_WEIGHT_MATMULS_2_FUSION_H_

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
class MultiWeightMatmulsFusion2 : public Pass {
 public:
  MultiWeightMatmulsFusion2() : Pass("multi_weight_matmuls_fusion2") {}
  ~MultiWeightMatmulsFusion2() override = default;
  bool Run(const FuncGraphPtr &graph) override;

 protected:
  void Process(const std::string &name, const AnfNodePtr &node, const AnfNodePtrList &users,
               AnfNodePtrList *getitems) const;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_MULTI_WEIGHT_MATMULS_FUSION_H_
