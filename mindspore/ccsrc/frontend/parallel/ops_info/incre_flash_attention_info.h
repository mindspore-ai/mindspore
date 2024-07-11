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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_INCRE_FLASH_ATTENTION_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_INCRE_FLASH_ATTENTION_INFO_H_

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
class IncreFlashAttentionInfo : public OperatorInfo {
 public:
  // Generate all strategies and the corresponding cost for the ifa operator
  IncreFlashAttentionInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                          const PrimitiveAttrs &attrs)
      : OperatorInfo(name, inputs_shape, outputs_shape, attrs, std::make_shared<MatMulCost>()) {}
  ~IncreFlashAttentionInfo() override = default;
  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;

  Status SetCostUnderStrategy(const StrategyPtr &strategy) override { return SetCostUnderStrategyBase(strategy); }
  void ReComputeBatchSplitFlagList() override;
  void ReplaceNodeInputOrAttrs() override;

 protected:
  Status InferForwardCommunication() override { return SUCCESS; }
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status GetAttrs() override;
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferMirrorOps() override;

 private:
  int64_t head_num_;
  int64_t kv_head_num;
  std::string input_layout_;
  int64_t dp_;
  int64_t mp_;
  std::vector<bool> optinal_inputs_;
  size_t atten_mask_rank_ = 0;
  size_t pse_shift_rank_ = 0;
  bool CheckStrategyOnIndex(int64_t strategy, int64_t true_value, const std::string &dim_name,
                            const std::string &input_name);
  std::vector<Shape> optinal_tensor_map_ = {
    {}, {}, {},      {1, -1, -1, -1}, {1},      {1},      {1, -1, -1, -1}, {},       {},       {},      {},
    {}, {}, {1, -1}, {-1, -1},        {-1, -1}, {-1, -1}, {-1, -1},        {-1, -1}, {-1, -1}, {-1, -1}};
  std::vector<Shape> optinal_op_strategies_ = {{},  {},  {},  {1, 0, 0, 0}, {1}, {1}, {1, 0, 0, 0},
                                               {0}, {0}, {0}, {0},          {0}, {0}, {1, 0},
                                               {0}, {0}, {0}, {0},          {0}, {0}, {0}};

  void SetOptinalInputs();
  size_t GetSqueezedIndex(size_t original_index);
  Status CheckAntiquantStrategy(const StrategyPtr &strategy, size_t input_index);
  Status CheckAttenMaskStrategy(const StrategyPtr &strategy, size_t input_index);
};
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_INCRE_FLASH_ATTENTION_INFO_H_
