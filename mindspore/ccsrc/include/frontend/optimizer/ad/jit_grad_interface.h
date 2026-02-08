/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_FRONTEND_OPTIMIZER_AD_JIT_GRAD_INTERFACE_H
#define MINDSPORE_CCSRC_INCLUDE_FRONTEND_OPTIMIZER_AD_JIT_GRAD_INTERFACE_H

#include <vector>
#include <string>
#include "ir/anf.h"
#include "pynative/utils/base.h"
#include "include/utils/visible.h"
#include "include/utils/pynative/variable.h"

namespace mindspore {
namespace ad {
constexpr auto kTopCellWithRecompute = "top_cell_with_recompute";
constexpr auto kOutputNoRecompute = "output_no_recompute";

FRONTEND_EXPORT std::pair<bool, FuncGraphPtr> GetBpropGraph(const pynative::GradParamPtr &grad_param);
FRONTEND_EXPORT void ClearGradCache();
FRONTEND_EXPORT std::pair<FuncGraphPtr, VectorRef> FilterGraph(const VectorRef &args, const VectorRef &added_args,
                                                               const FuncGraphPtr &func_graph,
                                                               const std::string &cache_key,
                                                               std::vector<pynative::autograd::Edge> *next_edges);
FRONTEND_EXPORT std::pair<FuncGraphPtr, FuncGraphPtr> CacheFuncGraphBeforeOpt(const FuncGraphPtr &jit_grad_graph,
                                                                              const FuncGraphPtr &jit_primal_graph);
}  // namespace ad
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_FRONTEND_OPTIMIZER_AD_JIT_GRAD_INTERFACE_H
