
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
#ifndef MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_ELIMINATE_MAKETUPLE_GETITEM_H_
#define MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_ELIMINATE_MAKETUPLE_GETITEM_H_

#include <vector>
#include "include/backend/optimizer/pass.h"
#include "ir/func_graph.h"
#include "backend/common/graph_kernel/core/graph_builder.h"

namespace mindspore::graphkernel {
/**
 * @brief Eliminate redundant MakeTuple-Getitem edges
 * @example
 *   %1 = op1
 *   %2 = op2
 *   %3 = make_tuple(%1, %2)
 *   %4 = tuple_getitem(%3, 0)
 *   %5 = tuple_getitem(%3, 1)
 *   %6 = op6(%4, %5)
 *   -->
 *   %1 = op1
 *   %2 = op2
 *   %6 = op6(%1, %2)
 */
class ElimMaketupleGetitem : public opt::Pass {
 public:
  ElimMaketupleGetitem() : Pass("elim_maketuple_getitem") {}
  ~ElimMaketupleGetitem() override = default;
  bool Run(const FuncGraphPtr &func_graph) override { return EliminateMaketupleGetitem(func_graph); }
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_ELIMINATE_MAKETUPLE_GETITEM_H_
