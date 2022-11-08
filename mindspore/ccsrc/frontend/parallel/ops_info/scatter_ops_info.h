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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_SCATTER_OPS_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_SCATTER_OPS_INFO_H_

#include <string>
#include <memory>
#include <vector>

#include "utils/hash_map.h"
#include "ir/value.h"
#include "frontend/parallel/auto_parallel/operator_costmodel.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/strategy.h"

namespace mindspore {
namespace parallel {
class ScatterOpsInfo : public OperatorInfo {
 public:
  ScatterOpsInfo(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                 const PrimitiveAttrs &attrs, const OperatorCostPtr &cost)
      : OperatorInfo(operator_name, inputs_shape, outputs_shape, attrs, cost) {}
  ~ScatterOpsInfo() override = default;

  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;
  Status SetCostUnderStrategy(const StrategyPtr &strategy) override;
  void ReComputeBatchSplitFlagList() override;

 protected:
  Status GetAttrs() override { return SUCCESS; }
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferMirrorOps() override { return SUCCESS; }  // the scatter_update only use in eval/predict
  Status InferForwardCommunication() override { return SUCCESS; }
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
};

class ScatterUpdateInfo : public ScatterOpsInfo {
 public:
  ScatterUpdateInfo(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                    const PrimitiveAttrs &attrs)
      : ScatterOpsInfo(operator_name, inputs_shape, outputs_shape, attrs, std::make_shared<ScatterUpdateCost>()) {}
  ~ScatterUpdateInfo() override = default;
};

using ScatterUpdateInfoPtr = std::shared_ptr<ScatterUpdateInfo>;

class ScatterMaxInfo : public ScatterOpsInfo {
 public:
  ScatterMaxInfo(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                 const PrimitiveAttrs &attrs)
      : ScatterOpsInfo(operator_name, inputs_shape, outputs_shape, attrs, std::make_shared<ScatterUpdateCost>()) {}
  ~ScatterMaxInfo() override = default;
};

using ScatterMaxInfoInfoPtr = std::shared_ptr<ScatterMaxInfo>;

class ScatterMinInfo : public ScatterOpsInfo {
 public:
  ScatterMinInfo(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                 const PrimitiveAttrs &attrs)
      : ScatterOpsInfo(operator_name, inputs_shape, outputs_shape, attrs, std::make_shared<ScatterUpdateCost>()) {}
  ~ScatterMinInfo() override = default;
};

using ScatterMinInfoInfoPtr = std::shared_ptr<ScatterMinInfo>;
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_SCATTER_OPS_INFO_H_
