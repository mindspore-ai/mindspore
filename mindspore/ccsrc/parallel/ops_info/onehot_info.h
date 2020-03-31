/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PARALLEL_OPS_INFO_ONEHOT_INFO_H_
#define MINDSPORE_CCSRC_PARALLEL_OPS_INFO_ONEHOT_INFO_H_

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

#include "ir/value.h"
#include "parallel/ops_info/operator_info.h"
#include "parallel/auto_parallel/operator_costmodel.h"
#include "parallel/strategy.h"

namespace mindspore {
namespace parallel {
class OneHotInfo : public OperatorInfo {
 public:
  OneHotInfo(const std::string& name, const Shapes& inputs_shape, const Shapes& outputs_shape,
             const PrimitiveAttrs& attrs)
      : OperatorInfo(name, inputs_shape, outputs_shape, attrs) {
    onehot_cost_ptr_ = std::make_shared<OneHotCost>();
  }
  ~OneHotInfo() override = default;
  Status Init(const StrategyPtr& strategy) override;
  Status InitForCostModel(const StrategyPtr& strategy) override;

  Status GenerateStrategies(int32_t stage_id) override;
  Status SetCostUnderStrategy(const StrategyPtr& strategy) override;
  OperatorCostPtr GetOperatorCost() const override { return onehot_cost_ptr_; }
  ReplaceGraphPtr replace_graph(const CNodePtr& cnode) override;
  std::shared_ptr<std::vector<std::vector<int32_t>>> GenerateBatchStrategies() override;

 protected:
  Status CheckStrategy(const StrategyPtr& strategy) override;
  Status GetAttrs() override;
  Status InferMirrorOps() override { return SUCCESS; }
  Status InferForwardCommunication() override { return SUCCESS; }
  Status InferTensorInfo() override;
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status ExtractInputInfo();

 private:
  Status ComputeReplaceGraph(const CNodePtr& cnode);

  int axis_ = -1;
  OneHotCostPtr onehot_cost_ptr_;
  int32_t rank_ = 0;
  int32_t total_class_number_ = 1;
  int32_t classes_each_device_ = 1;
  ValuePtr axis_value_ptr_;
  int32_t mod_rank_ = 0;
};
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_OPTIMIZER_OPS_INFO_PARALLEL_ONEHOT_INFO_H_
