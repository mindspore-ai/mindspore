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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_BATCH_PARALLEL_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_BATCH_PARALLEL_INFO_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "ir/value.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/strategy.h"

namespace mindspore {
namespace parallel {
class BatchParallelInfo : public OperatorInfo {
 public:
  BatchParallelInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                    const PrimitiveAttrs &attrs, OperatorCostPtr cost)
      : OperatorInfo(name, inputs_shape, outputs_shape, attrs, cost), dev_num_(1) {}
  BatchParallelInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                    const PrimitiveAttrs &attrs)
      : OperatorInfo(name, inputs_shape, outputs_shape, attrs, std::make_shared<BatchParallelCost>()), dev_num_(1) {}

  ~BatchParallelInfo() override = default;
  Status Init(const StrategyPtr &strategy) override;
  Status InitForCostModel(const StrategyPtr &strategy) override;
  Status GenerateStrategies(int64_t stage_id) override;
  Status SetCostUnderStrategy(const StrategyPtr &strategy) override;

 protected:
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferForwardCommunication() override;
  Status InferTensorInfo() override;
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status GetAttrs() override;
  Strategys GetOutputsStrategy();
  Status InferAsLossDivisor() override;

 private:
  int64_t dev_num_;
};

class SparseSoftmaxCrossEntropyWithLogitsInfo : public BatchParallelInfo {
 public:
  SparseSoftmaxCrossEntropyWithLogitsInfo(const std::string &name, const Shapes &inputs_shape,
                                          const Shapes &outputs_shape, const PrimitiveAttrs &attrs)
      : BatchParallelInfo(name, inputs_shape, outputs_shape, attrs,
                          std::make_shared<SparseSoftmaxCrossEntropyWithLogitsCost>()) {}
  ~SparseSoftmaxCrossEntropyWithLogitsInfo() override = default;
  void ReComputeBatchSplitFlagList() override;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_BATCH_PARALLEL_INFO_H_
