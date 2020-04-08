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

#ifndef MINDSPORE_CCSRC_PARALLEL_OPS_INFO_DROPOUT_DO_MASK_INFO_H_
#define MINDSPORE_CCSRC_PARALLEL_OPS_INFO_DROPOUT_DO_MASK_INFO_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "ir/value.h"
#include "parallel/auto_parallel/operator_costmodel.h"
#include "parallel/ops_info/operator_info.h"
#include "parallel/strategy.h"

namespace mindspore {
namespace parallel {
class DropoutDoMaskInfo : public OperatorInfo {
 public:
  DropoutDoMaskInfo(const std::string& name, const Shapes& inputs_shape, const Shapes& outputs_shape,
                    const PrimitiveAttrs& attrs)
      : OperatorInfo(name, inputs_shape, outputs_shape, attrs) {
    bpcost_ptr_ = std::make_shared<BatchParallelCost>();
  }
  ~DropoutDoMaskInfo() override = default;

  Status Init(const StrategyPtr& strategy) override;
  Status GenerateStrategies(int32_t stage_id) override;
  Status SetCostUnderStrategy(const StrategyPtr& strategy) override;
  OperatorCostPtr GetOperatorCost() const override { return bpcost_ptr_; }
  Status InitForCostModel(const StrategyPtr& strategy) override;
  std::shared_ptr<std::vector<std::vector<int32_t>>> GenerateBatchStrategies() override;

 protected:
  Status CheckStrategy(const StrategyPtr& strategy) override;
  Status InferMirrorOps() override { return SUCCESS; }
  Status InferForwardCommunication() override { return SUCCESS; }
  Status InferTensorMap() override;
  Status GetAttrs() override { return SUCCESS; }
  Status InferTensorInfo() override;
  Status InferDevMatrixShape() override;

 private:
  BatchParallelCostPtr bpcost_ptr_;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PARALLEL_OPS_INFO_DROPOUT_DO_MASK_INFO_H_
