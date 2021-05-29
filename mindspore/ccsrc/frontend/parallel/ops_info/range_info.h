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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_RANGE_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_RANGE_INFO_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "utils/ms_utils.h"
#include "ir/value.h"
#include "frontend/parallel/auto_parallel/operator_costmodel.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/strategy.h"

namespace mindspore {
namespace parallel {
// Range op:
//    (start=8.0, limit=16.0, delta=1.0) -> [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
//    (start=8.0, limit=None, delta=1.0) -> [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
// when entering the step_parallel, the limit=None has been processed
// the parallel op need to modify the 'start' and 'limit'
class RangeInfo : public OperatorInfo {
 public:
  RangeInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
            const PrimitiveAttrs &attrs)
      : OperatorInfo(name, inputs_shape, outputs_shape, attrs, std::make_shared<RangeCost>()) {}
  ~RangeInfo() override = default;

  Status Init(const StrategyPtr &strategy) override;
  Status InitForCostModel(const StrategyPtr &strategy) override;

  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;
  Status SetCostUnderStrategy(const StrategyPtr &strategy) override;

 protected:
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferMirrorOps() override;
  Status InferForwardCommunication() override;
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status GetAttrs() override;
  float GetRangeAttr(const std::string &arg);

  float start_ = 0.0;
  float limit_ = 0.0;
  float delta_ = 0.0;
  float new_start_ = 0.0;
  float new_limit_ = 0.0;
  int64_t split_num_ = 1;
};
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_RANGE_INFO_H_
