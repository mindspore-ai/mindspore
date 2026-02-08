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
#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SHRINK_ONLY_SHAPE_NEEDED_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SHRINK_ONLY_SHAPE_NEEDED_H_

#include "include/backend/common/pass_manager/pass.h"

namespace mindspore::graphkernel {
/**
 * @brief Replace sub graph output with sub graph input if this output is only used by shape op and
 *        the shape of output and input are equal.
 * @example
 *   sub_graph(p0, p1) {
 *     %0 = Op1(p0)  // p0 and %0 has same shape
 *     %1 = Op2(%0)
 *     return %0, %1
 *   }
 *
 *   main_graph {
 *     %0 = sub_graph(p0, p1)
 *     %1 = tuple_getitem(%0, 0)
 *     %2 = tuple_getitem(%0, 1)
 *     %3 = Shape(%1)
 *     %4 = Op(p2, %3)
 *     ...
 *   }
 *   ---------->
 *   sub_graph(p0, p1) {
 *     %0 = Op1(p0)
 *     %1 = Op2(%0)
 *     return %1
 *   }
 *
 *   main_graph {
 *     %0 = sub_graph(p0, p1)
 *     %1 = Shape(p0)
 *     %2 = Op(p2, %1)
 *     ...
 *   }
 */
class ShrinkOnlyShapeNeeded : public opt::Pass {
 public:
  ShrinkOnlyShapeNeeded() : Pass("shrink_only_shape_needed") {}
  ~ShrinkOnlyShapeNeeded() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  bool Process(const AnfNodePtr &node, const FuncGraphManagerPtr &mng) const;
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SHRINK_ONLY_SHAPE_NEEDED_H_
