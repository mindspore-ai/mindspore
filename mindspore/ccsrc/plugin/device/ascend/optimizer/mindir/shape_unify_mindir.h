/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_SHAPE_UNIFY_MINDIR_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_SHAPE_UNIFY_MINDIR_H_

#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class ShapeUnifyMindIR : public PatternProcessPass {
 public:
  explicit ShapeUnifyMindIR(bool multigraph = true) : PatternProcessPass("shape_unify_mindir", multigraph) {
    is_add_ = false;
  }
  ~ShapeUnifyMindIR() override = default;

  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  ValueNodePtr CreateScalarValueTuple(const FuncGraphPtr &func_graph, int64_t value) const;
  CNodePtr CreateTensorToScalar(const FuncGraphPtr &func_graph, const AnfNodePtr &anf_node) const;
  CNodePtr CreateTensorShape(const FuncGraphPtr &func_graph, const AnfNodePtr &anf_node) const;
  CNodePtr CreateStridedSlice(const FuncGraphPtr &func_graph, const AnfNodePtr &shape_node,
                              const AnfNodePtr &tuple_get_node, const FuncGraphManagerPtr &manager) const;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_SHAPE_UNIFY_MINDIR_H_
