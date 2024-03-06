/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_APPLY_ROTARY_POS_EMB_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_APPLY_ROTARY_POS_EMB_INFO_H_

#include <memory>
#include <string>
#include <vector>

#include "utils/hash_map.h"
#include "ir/value.h"
#include "frontend/parallel/auto_parallel/operator_costmodel.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/strategy.h"

namespace mindspore {
namespace parallel {
constexpr size_t kApplyRotaryOutputSize = 2;
constexpr size_t kApplyRotaryPosEmbQueryIndex = 0;
constexpr size_t kApplyRotaryPosEmbKeyIndex = 1;
constexpr size_t kApplyRotaryPosEmbCosIndex = 2;
constexpr size_t kApplyRotaryPosEmbSinIndex = 3;
constexpr size_t kApplyRotaryPosEmbPositionIdsIndex = 4;
constexpr size_t kInputQueryBatchIndex = 0;
constexpr size_t kInputQuerySeqLenIndex = 1;
constexpr size_t kInputQueryHiddenSizeIndex = 2;
constexpr size_t kInputKeyBatchIndex = 0;
constexpr size_t kInputKeySeqLenIndex = 1;
constexpr size_t kInputKeyHiddenSizeIndex = 2;
constexpr size_t kInputCosSeqLenIndex = 0;
constexpr size_t kInputCosHeaDimIndex = 1;
constexpr size_t kInputSinSeqLenIndex = 0;
constexpr size_t kInputSinHeaDimIndex = 1;
constexpr size_t kInputPositionIdsBatchIndex = 2;
constexpr size_t kIncInferSeqLen = 1;
class ApplyRotaryPosEmbInfo : public OperatorInfo {
 public:
  ApplyRotaryPosEmbInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                        const PrimitiveAttrs &attrs)
      : OperatorInfo(name, inputs_shape, outputs_shape, attrs, std::make_shared<ActivationInfoCost>()) {}
  ~ApplyRotaryPosEmbInfo() override = default;
  Status CheckStrategy(const StrategyPtr &strategy) override;
  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override { return {}; }
  Status SetCostUnderStrategy(const StrategyPtr &strategy) override { return SetCostUnderStrategyBase(strategy); }

 protected:
  Status GetAttrs() override { return SUCCESS; }
  Status InferForwardCommunication() { return SUCCESS; }
  Status InferTensorMap() override;
  Status InferDevMatrixShape() override;
  Status InferAsLossDivisor() override;
};
using ApplyRotaryPosEmbInfoPtr = std::shared_ptr<ApplyRotaryPosEmbInfo>;
}  // namespace parallel
}  // namespace mindspore

#endif  // MMINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_APPLY_ROTARY_POS_EMB_INFO_H_
