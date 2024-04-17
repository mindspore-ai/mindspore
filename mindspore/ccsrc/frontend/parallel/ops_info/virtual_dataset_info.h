/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#ifndef PARALLEL_OPS_INFO_DATASET_INFO_H_
#define PARALLEL_OPS_INFO_DATASET_INFO_H_

#include <memory>
#include <string>
#include <vector>

#include "utils/hash_map.h"
#include "ir/value.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/strategy.h"

namespace mindspore {
namespace parallel {
class VirtualDatasetInfo : public OperatorInfo {
 public:
  VirtualDatasetInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                     const PrimitiveAttrs &attrs)
      : OperatorInfo(name, inputs_shape, outputs_shape, attrs, std::make_shared<VirtualDatasetCost>()) {}
  ~VirtualDatasetInfo() override = default;
  Status Init(const StrategyPtr &in_strategy, const StrategyPtr &out_strategy,
              const std::vector<std::shared_ptr<TensorLayout>> &in_tensor_layouts = {},
              const std::vector<std::shared_ptr<TensorLayout>> &out_tensor_layouts = {}) override;
  Status InitForCostModel(const StrategyPtr &in_strategy, const StrategyPtr &out_strategy) override;

  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;
  Status SetCostUnderStrategy(const StrategyPtr &strategy) override;
  void ReComputeBatchSplitFlagList() override;

 protected:
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferMirrorOps() override;
  Status InferForwardCommunication() override;
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status GetAttrs() override;
  Status InferAsLossDivisor() override;
  size_t max_size_strategy_dim_ = 0;
  int64_t shard_num_ = 1;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // PARALLEL_OPS_INFO_VIRTUAL_DATASET_INFO_H_
