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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_GRAD_VAR_PREPARE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_GRAD_VAR_PREPARE_H_

#include <vector>
#include <algorithm>
#include <unordered_map>
#include <memory>

#include "frontend/operator/composite/composite.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/anf_visitor.h"
#include "ir/func_graph.h"
#include "ir/func_graph_cloner.h"

namespace mindspore {
namespace opt {
namespace irpass {
// {{GradOperation, g, w}, Ys}
// {UnPackCall, {GradOperation, g, w}, Ys}
class GradVarPrepare : public AnfVisitor {
 public:
  GradVarPrepare()
      : grad_op_(std::make_shared<prim::GradOperation>("grad")),
        unpack_op_(std::make_shared<prim::UnpackCall>("unpack_call")) {}
  ~GradVarPrepare() override = default;

  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;

 private:
  MetaFuncGraphPtr grad_op_;
  MetaFuncGraphPtr unpack_op_;
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_GRAD_VAR_PREPARE_H_
