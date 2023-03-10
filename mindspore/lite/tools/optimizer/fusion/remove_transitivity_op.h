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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_REMOVE_TRANSITIVITY_OP_H
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_REMOVE_TRANSITIVITY_OP_H

#include "include/backend/optimizer/pass.h"
#include "tools/optimizer/graph/preprocess_dynamic_shape.h"

namespace mindspore {
namespace opt {
// remove the op whose output is equal to its input.
class RemoveTransitivityOp : public Pass {
 public:
  RemoveTransitivityOp() : Pass("RemoveTransitivityOp") {}
  ~RemoveTransitivityOp() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  int HandleStridedSlice(const FuncGraphPtr &func_graph, const CNodePtr &strided_slice);
  int HandleConcat(const FuncGraphPtr &func_graph, const CNodePtr &concat);
  int HandleReduce(const FuncGraphPtr &func_graph, const CNodePtr &reduce);
  int DoReplace(const FuncGraphPtr &func_graph, const CNodePtr &cnode);
  DynamicShapePreprocessor preprocessor_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_REMOVE_TRANSITIVITY_OP_H
