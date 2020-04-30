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

#ifndef MINDSPORE_CCSRC_PARALLEL_OPS_INFO_GATHER_V2_P_INFO_H_
#define MINDSPORE_CCSRC_PARALLEL_OPS_INFO_GATHER_V2_P_INFO_H_

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
class GatherV2PInfo : public OperatorInfo {
 public:
  GatherV2PInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                const PrimitiveAttrs &attrs)
      : OperatorInfo(name, inputs_shape, outputs_shape, attrs, std::make_shared<GatherV2PCost>()) {}
  ~GatherV2PInfo() override = default;
  Status Init(const StrategyPtr &strategy) override;
  Status InitForCostModel(const StrategyPtr &strategy) override;

  Status GenerateStrategies(int32_t stage_id) override;
  Status SetCostUnderStrategy(const StrategyPtr &strategy) override;
  std::shared_ptr<std::vector<std::vector<int32_t>>> GenerateBatchStrategies() override;

 protected:
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferMirrorOps() override { return SUCCESS; }
  Status InferForwardCommunication() override { return SUCCESS; }
  Status InferTensorInfo() override;
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status GetAttrs() override;

 private:
  Status ComputeReplaceGraph(const CNodePtr &cnode);
  Status InferBias();
  Status InferGroup();

  int32_t axis_;
  int32_t bias_;
  int32_t slice_size_;
  Shape out_dev_matrix_shape_;
  Group group_;
};
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PARALLEL_OPS_INFO_GATHER_V2_P_INFO_H_
