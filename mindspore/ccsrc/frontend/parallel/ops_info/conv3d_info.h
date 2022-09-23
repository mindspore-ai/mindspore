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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_CONV3D_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_CONV3D_INFO_H_

#include <string>
#include <memory>
#include <vector>

#include "utils/hash_map.h"
#include "ir/value.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/auto_parallel/operator_costmodel.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/ops_info/conv2d_info.h"
#include "frontend/parallel/strategy.h"

namespace mindspore {
namespace parallel {
class Conv3DInfo : public Conv2DInfo {
 public:
  Conv3DInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
             const PrimitiveAttrs &attrs)
      : Conv2DInfo(name, inputs_shape, outputs_shape, attrs) {}
  ~Conv3DInfo() override = default;

 protected:
  Status CheckAttrsBase() override;
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferTensorMap() override;
  std::string ReplaceNodeName() const override;
  AnfNodePtr GenerateConv3DNode(const AnfNodePtr &new_input, const CNodePtr &cnode);
  void ComputeReplaceGraph(const CNodePtr &cnode) override;
  std::vector<int64_t> GetStrideAttr() override;
  std::vector<int64_t> GetDilationAttr() override;
  OperatorAttrs CreateConv3DAttrs();
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_CONV3D_INFO_H_
