/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_DSD_MATMUL_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_DSD_MATMUL_INFO_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "ir/value.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/strategy.h"

namespace mindspore {
namespace parallel {
/*
 * parallel class for MatmulDDS Primitive
 */
class DSDMatmulInfo : public OperatorInfo {
 public:
  DSDMatmulInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                const PrimitiveAttrs &attrs)
      : OperatorInfo(name, inputs_shape, outputs_shape, attrs, std::make_shared<DSDMatmulCost>()) {}
  ~DSDMatmulInfo() override = default;
  Status Init(const StrategyPtr &strategy) override;
  Status InitForCostModel(const StrategyPtr &strategy) override;

  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;
  Status SetCostUnderStrategy(const StrategyPtr &strategy) override;

 protected:
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferForwardCommunication() override { return SUCCESS; }
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status GetAttrs() override;
  Status InferAsLossDivisor() override { return SUCCESS; }
  Status ComputeReplaceGraph(const CNodePtr &cnode);

 private:
  Dimensions input_strategy_;
  int64_t batch_size_ = 0;
  int64_t num_heads_ = 0;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_DSD_MATMUL_INFO_H_
