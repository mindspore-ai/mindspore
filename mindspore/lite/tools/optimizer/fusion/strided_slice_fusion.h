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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_STRIDED_SLICE_FUSION_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_STRIDED_SLICE_FUSION_H_
#include <vector>
#include "include/backend/optimizer/pass.h"

namespace mindspore {
namespace opt {
class StridedSliceFusion : public Pass {
 public:
  StridedSliceFusion() : Pass("StridedSliceFusionInHorizon") {}
  ~StridedSliceFusion() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  int Process(const FuncGraphPtr &func_graph, const CNodePtr &cnode);
  void FindStridedSliceOp(const FuncGraphPtr &func_graph, const CNodePtr &cnode);
  bool CheckCanFusion();
  std::vector<std::vector<CNodePtr>> strided_slice_ops_;
  int axis_{0};
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_STRIDED_SLICE_FUSION_H_
