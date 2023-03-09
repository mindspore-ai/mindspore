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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_REDUCE_STACK_FUSION_H
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_REDUCE_STACK_FUSION_H

#include "ir/anf.h"
#include "include/backend/optimizer/pass.h"

namespace mindspore {
namespace opt {
class ReduceStackFusion : public Pass {
 public:
  ReduceStackFusion() : Pass("ReduceStackFusion") {}
  ~ReduceStackFusion() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  int Process(const FuncGraphPtr &func_graph, const CNodePtr &stack);
  bool CheckCanFusion(const FuncGraphPtr &func_graph, const CNodePtr &stack);
  bool CheckReduce(const FuncGraphPtr &func_graph, const CNodePtr &reduce, int stack_axis);
  PrimitivePtr reduce_prim_{nullptr};
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_REDUCE_STACK_FUSION_H
