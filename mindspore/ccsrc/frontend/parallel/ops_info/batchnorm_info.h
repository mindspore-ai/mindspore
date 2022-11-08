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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_BATCHNORM_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_BATCHNORM_INFO_H_

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
class BatchNormInfo : public OperatorInfo {
 public:
  BatchNormInfo(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                const PrimitiveAttrs &attrs)
      : OperatorInfo(operator_name, inputs_shape, outputs_shape, attrs, std::make_shared<BatchNormCost>()) {}
  ~BatchNormInfo() override = default;

  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;
  Status SetCostUnderStrategy(const StrategyPtr &strategy) override;

 protected:
  Status GetAttrs() override;
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferForwardCommunication() override;
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  void InferReplaceOps() override;
  Status InferAsLossDivisor() override;

 private:
  bool is_training_ = false;
  float epsilon_ = 0.00001;
  float momentum_ = 0.1;
  bool input_is_4d_ = true;
  int64_t group_size_ = -1;  // config the allreduce group size for SyncBatchnorm, it only support that sharding n dim
  std::string format_;
  std::vector<Group> forward_allreduce_group_;
  Status InferAllReduceGroupBySize();
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_BATCHNORM_INFO_H_
