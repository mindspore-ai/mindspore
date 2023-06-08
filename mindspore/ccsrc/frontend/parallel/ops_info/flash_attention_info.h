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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_FLASH_ATTENTION_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_FLASH_ATTENTION_INFO_H_

#include <memory>
#include <string>
#include <vector>

#include "utils/hash_map.h"
#include "utils/ms_utils.h"
#include "ir/value.h"
#include "frontend/parallel/auto_parallel/operator_costmodel.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/strategy.h"

namespace mindspore {
namespace parallel {
class FlashAttentionPrimitiveInfo : public OperatorInfo {
 public:
  // Generate all strategies and the corresponding cost for this MatMul operator
  FlashAttentionPrimitiveInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                              const PrimitiveAttrs &attrs)
      : OperatorInfo(name, inputs_shape, outputs_shape, attrs, std::make_shared<MatMulCost>()) {}
  ~FlashAttentionPrimitiveInfo() override = default;
  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;

  Status SetCostUnderStrategy(const StrategyPtr &strategy) override;

 protected:
  Status InferForwardCommunication() override { return SUCCESS; }
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status GetAttrs() override;
  Status InferAsLossDivisor() override;
  Status InferMirrorOps() override;
  Status CheckStrategy(const StrategyPtr &strategy) override;
  size_t qkv_dim_sz = 4;
  size_t dim_mask_dim_sz = 1;
  size_t attn_mask_dim_sz = 3;
  size_t dropout_mask_dim_sz = 4;
  size_t alibi_mask_dim_sz = 4;

  bool alibi_mask_valid = false;
  constexpr static int DROPOUT_MASK_INPUT_INDEX = 5;
};
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_FLASH_ATTENTION_INFO_H_
