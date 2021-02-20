/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_AD_KPYNATIVE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_AD_KPYNATIVE_H_

#include <memory>

#include "ir/anf.h"
#include "ir/func_graph.h"

namespace mindspore {
namespace ad {
class KPynativeCell {
 public:
  virtual ~KPynativeCell() = default;
};

using KPynativeCellPtr = std::shared_ptr<KPynativeCell>;

FuncGraphPtr OptimizeBPropFuncGraph(const FuncGraphPtr &bprop_fg, const CNodePtr &c_node, const ValuePtrList &op_args,
                                    const ValuePtr &out);

KPynativeCellPtr GradPynativeCellBegin(const AnfNodePtrList &cell_inputs);
FuncGraphPtr GradPynativeCellEnd(const KPynativeCellPtr &k_cell, const AnfNodePtrList &weights, bool grad_inputs,
                                 bool grad_weights);
bool GradPynativeOp(const KPynativeCellPtr &k_cell, const CNodePtr &c_node, const ValuePtrList &op_args,
                    const ValuePtr &out);
}  // namespace ad
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_AD_GRAD_H_
