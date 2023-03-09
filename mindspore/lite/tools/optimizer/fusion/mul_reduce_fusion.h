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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_MUL_REDUCE_FUSION_H
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_MUL_REDUCE_FUSION_H

#include <map>
#include <string>
#include <utility>
#include "include/backend/optimizer/pass.h"
#include "tools/optimizer/graph/preprocess_dynamic_shape.h"

namespace mindspore {
namespace opt {
class MulReduceFusion : public Pass {
 public:
  explicit MulReduceFusion(const std::string &name = "MulReduceFusion") : Pass(name) {}
  ~MulReduceFusion() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  int ProcessOp(const FuncGraphPtr &func_graph, const CNodePtr &cnode);
  int PostProcess(const FuncGraphPtr &func_graph);
  int PostProcessSqueezeWithConcat(const FuncGraphPtr &func_graph, const CNodePtr &cnode);
  int GenerateMatmul(const FuncGraphPtr &func_graph, const CNodePtr &cnode);
  int GenerateSqueeze(const FuncGraphPtr &func_graph, const CNodePtr &cnode);
  int GenerateMul(const FuncGraphPtr &func_graph, const CNodePtr &cnode);
  int ProcessGather();
  bool CheckBasicCond(const FuncGraphPtr &func_graph, const CNodePtr &cnode);
  bool CheckAxisCond(const CNodePtr &cnode);
  bool CheckShapeCond(const CNodePtr &cnode);
  bool CheckGatherOp(const FuncGraphPtr &func_graph, const CNodePtr &cnode);
  bool CheckConcatOp(const FuncGraphPtr &func_graph, const CNodePtr &cnode);
  bool exchange_{false};      // determine if exchange the two inputs of mul.
  bool transpose_a_{false};   // determine matmul a-matrix's attr.
  bool transpose_b_{false};   // determine matmul b-matrix's attr.
  bool keep_dim_{false};      // record the keep-dim attr of reduce.
  int axis_{0};               // record the axis of reduce.
  int reduce_mode_{0};        // record the reduce_mode of reduce.
  float coeff_{1.0f};         // valid when reduce_mode_ is reduce_mean, we can break it down into reduce_sum * coeff_.
  int concat_axis_{0};        // record the new axis for concat-op in PostProcess.
  CNodePtr gather_{nullptr};  // gather's first input, valid when reduce_mode_ is reduce_mean.
  DynamicShapePreprocessor preprocessor_;
  std::map<CNodePtr, std::pair<int, int>>
    squeeze_infos_;  // record generated-squeeze(<op, <axis, out-dims>>) which is used to post-fusion.
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_MUL_REDUCE_FUSION_H
