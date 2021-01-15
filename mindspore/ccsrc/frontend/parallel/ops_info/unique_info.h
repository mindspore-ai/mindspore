/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_UNIQUE_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_UNIQUE_INFO_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "frontend/parallel/auto_parallel/operator_costmodel.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/strategy.h"

namespace mindspore {
namespace parallel {
class UniqueInfo : public OperatorInfo {
 public:
  UniqueInfo(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
             const PrimitiveAttrs &attrs)
      : OperatorInfo(operator_name, inputs_shape, outputs_shape, attrs, std::make_shared<UniqueCost>()) {}
  ~UniqueInfo() override = default;

  Status Init(const StrategyPtr &strategy) override;
  Status SetCostUnderStrategy(const StrategyPtr &strategy) override;
  Status InitForCostModel(const StrategyPtr &strategy) override;
  Status GenerateStrategies(int64_t stage_id) override;
  ReplaceGraphPtr replace_graph(const CNodePtr &cnode) override;

 protected:
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status GetAttrs() override;
  Status InferTensorMap() override;
  Status InferTensorLayout(TensorLayouts *inputs_layout, TensorLayouts *outputs_layout);
  Status InferTensorInfo() override;
  Status InferDevMatrixShape() override;
  Status InferMirrorOps() override;
  Status InferForwardCommunication() override { return SUCCESS; }
  Status InferAsLossDivisor() override { return SUCCESS; }
#if (ENABLE_CPU && (ENABLE_D || ENABLE_GPU))
  Status ComputeReplaceGraph(const CNodePtr &cnode);
#endif

 private:
  std::string replace_op_name_ = UNIQUE;
  int64_t dev_num_ = 1;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_UNIQUE_INFO_H_
