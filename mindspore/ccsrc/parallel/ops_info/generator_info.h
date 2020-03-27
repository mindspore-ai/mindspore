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

#ifndef MINDSPORE_CCSRC_PARALLEL_OPS_INFO_GENERATOR_INFO_H_
#define MINDSPORE_CCSRC_PARALLEL_OPS_INFO_GENERATOR_INFO_H_

#include <string>
#include <list>
#include <unordered_map>
#include <vector>
#include <memory>

#include "parallel/ops_info/operator_info.h"
#include "parallel/auto_parallel/operator_costmodel.h"
#include "parallel/strategy.h"

namespace mindspore {
namespace parallel {
class GeneratorBase : public OperatorInfo {
 public:
  GeneratorBase(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                const PrimitiveAttrs &attrs)
      : OperatorInfo(operator_name, inputs_shape, outputs_shape, attrs) {
    generatorbasecost_ptr_ = std::make_shared<GeneratorBaseCost>();
  }

  ~GeneratorBase() override = default;

  Status Init(const StrategyPtr &strategy) override;
  Status SetCostUnderStrategy(const StrategyPtr &strategy) override;
  OperatorCostPtr GetOperatorCost() const override { return generatorbasecost_ptr_; }
  Status InitForCostModel(const StrategyPtr &strategy) override;

 protected:
  // For now, generator ops don't have attributes
  Status GetAttrs() override { return Status::SUCCESS; }
  Status InferTensorMap() override;
  Status InferTensorInfo() override;
  Status InferDevMatrixShape() override;
  Status InferMirrorOps() override { return SUCCESS; }
  Status InferForwardCommunication() override { return SUCCESS; }
  virtual Status InferReplaceOps(const StrategyPtr &strategy) = 0;
  GeneratorBaseCostPtr generatorbasecost_ptr_;
};

class DropoutGenMaskInfo : public GeneratorBase {
 public:
  DropoutGenMaskInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                     const PrimitiveAttrs &attrs)
      : GeneratorBase(name, inputs_shape, outputs_shape, attrs) {}
  ~DropoutGenMaskInfo() override = default;
  Status GenerateStrategies(int32_t stage_id) override;
  std::shared_ptr<std::vector<std::vector<int32_t>>> GenerateBatchStrategies() override;

 protected:
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferReplaceOps(const StrategyPtr &strategy) override;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PARALLEL_OPS_INFO_GENERATOR_INFO_H_
