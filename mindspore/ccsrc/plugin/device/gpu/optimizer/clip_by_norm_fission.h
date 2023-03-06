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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_CLIP_BY_NORM_FISSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_CLIP_BY_NORM_FISSION_H_

#include <memory>
#include <vector>
#include <string>
#include "backend/common/optimizer/optimizer.h"
#include "backend/common/optimizer/pattern_engine.h"

// This pass will split `ClipByNorm` op to smaller ops, such as `square`, `sqrt`, `reducesum` to achieve same function
namespace mindspore {
namespace opt {
class ClipByNormFission : public PatternProcessPass {
 public:
  explicit ClipByNormFission(bool multigraph = true) : PatternProcessPass("clip_by_norm_fission", multigraph) {}
  ~ClipByNormFission() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  AnfNodePtr CreateCNodeBase(const FuncGraphPtr &func_graph, const std::vector<AnfNodePtr> &inps,
                             const std::string &op_name, const AnfNodePtr &node) const;
  AnfNodePtr CreateCastNode(const FuncGraphPtr &func_graph, const AnfNodePtr &inp, const ShapeVector &shape_vec,
                            const TypeId &src_type_id, const TypeId &dst_type_id) const;
  AnfNodePtr CreateSquareNode(const FuncGraphPtr &func_graph, const AnfNodePtr &inp, const ShapeVector &shape_vec,
                              const TypeId &type_id) const;
  AnfNodePtr CreateReduceSumNode(const FuncGraphPtr &func_graph, const AnfNodePtr &square,
                                 const AnfNodePtr &clip_by_norm, const ShapeVector &shape_vec,
                                 const TypeId &type_id) const;
  AnfNodePtr CreateConstantNode(const FuncGraphPtr &func_graph, const AnfNodePtr &inp, const ShapeVector &shape_vec,
                                const TypeId &type_id, const std::string &op_name) const;
  AnfNodePtr CreateGreaterNode(const FuncGraphPtr &func_graph, const AnfNodePtr &inp_a, const AnfNodePtr &inp_b,
                               const ShapeVector &shape_vec) const;
  AnfNodePtr CreateSelectNode(const FuncGraphPtr &func_graph, const AnfNodePtr &cond, const AnfNodePtr &inp_a,
                              const AnfNodePtr &inp_b, const ShapeVector &shape_vec, const TypeId &type_id) const;
  AnfNodePtr CreateSqrtNode(const FuncGraphPtr &func_graph, const AnfNodePtr &reduce_sum, const TypeId &type_id) const;
  AnfNodePtr CreateMulNode(const FuncGraphPtr &func_graph, const AnfNodePtr &x, const AnfNodePtr &clip_norm,
                           const ShapeVector &shape_vec, const TypeId &type_id) const;
  AnfNodePtr CreateMaxNode(const FuncGraphPtr &func_graph, const AnfNodePtr &x, const AnfNodePtr &y,
                           const TypeId &type_id) const;
  AnfNodePtr CreateDivNode(const FuncGraphPtr &func_graph, const AnfNodePtr &dividend, const AnfNodePtr &divisor,
                           const ShapeVector &shape_vec, const TypeId &type_id) const;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_CLIP_BY_NORM_FISSION_H_
