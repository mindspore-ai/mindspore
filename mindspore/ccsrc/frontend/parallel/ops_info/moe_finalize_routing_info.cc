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

#include "frontend/parallel/ops_info/moe_finalize_routing_info.h"
#include "frontend/parallel/dynamic_creator.h"

namespace mindspore {
namespace parallel {
// MoeFinalizeRouting has 7 inputs and 1 outputs (token_num is  bs*seq)
// expanded_x:            2D Tensor (token_num * top_k, hidden)
// skip1:                 2D Tensor (token_num, hidden)
// skip2(optional):       2D Tensor (token_num, hidden)
// bias:                  2D Tensor (expert_num, hidden)
// scales:                2D Tensor (token_num, top_k)
// expanded_row_idx:      1D Tensor (token_num * top_k)
// expanded_expert_idx:   2D Tensor (token_num, top_k)
// ------------------------------
// y:                     2D Tensor (token_num, hidden)

// split strategy
// [token_num, expert_num, hidden]
// k is not able to split

constexpr size_t minInputNum = 6;
constexpr size_t maxInputNum = 7;
constexpr size_t kexpandedx = 0;
constexpr size_t kskip1 = 1;
constexpr size_t kskip2opt = 2;
constexpr size_t kbias = 3;
constexpr size_t kscales = 4;
constexpr size_t krowIdx = 5;
constexpr size_t kexpertIdx = 6;

Status MoeFinalizeRoutingInfo::GetInputNumsAndGetIdx(const StrategyPtr &strategy) {
  auto input_nums = strategy->GetInputDim().size();
  if (input_nums != minInputNum && input_nums != maxInputNum) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: the input nums must be 6 or 7. But current input nums is "
                  << input_nums;
    return FAILED;
  }
  int64_t needDecrease = 0;
  if (input_nums == minInputNum) {
    needDecrease = 1;  // if kskip2 is not provide, idx after kskip2opt need decrease 1
  }
  input_nums_ = input_nums;

  // input idx
  kexpandedx_ = kexpandedx;                 // 0
  kskip1_ = kskip1;                         // 1
  kskip2opt_ = kskip2opt;                   // 2 or not use
  kbias_ = kbias - needDecrease;            // 3 or 2
  kscales_ = kscales - needDecrease;        // 4 or 3
  krowIdx_ = krowIdx - needDecrease;        // 5 or 4
  kexpertIdx_ = kexpertIdx - needDecrease;  // 6 or 5
  return SUCCESS;
}

Status MoeFinalizeRoutingInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }
  if (GetInputNumsAndGetIdx(strategy) != SUCCESS) {
    return FAILED;
  }

  auto input_strategys = strategy->GetInputDim();
  auto strategy_expanded_x = input_strategys.at(kexpandedx_);  // (kN, h)
  auto strategy_skpi1 = input_strategys.at(kskip1_);           // (N,  h)
  auto strategy_bias = input_strategys.at(kbias_);             // (E,  h)
  auto strategy_scales = input_strategys.at(kscales_);         // (N,  k)
  auto strategy_rowidx = input_strategys.at(krowIdx_);         // (kN)
  auto strategy_expertidx = input_strategys.at(kexpertIdx_);   // (N,  k)

  // token_num = N
  if (strategy_expanded_x.at(0) != strategy_skpi1.at(0) || strategy_expanded_x.at(0) != strategy_rowidx.at(0) ||
      strategy_expanded_x.at(0) != strategy_scales.at(0) || strategy_expanded_x.at(0) != strategy_expertidx.at(0)) {
    MS_LOG(ERROR) << name_
                  << ": Invalid strategy: token_num(strategy_expanded_x) == token_num(skip1) == "
                     "token_num(expanded_row_idx) == token_num(strategy_scales) == token_num(strategy_expertidx)."
                  << " But current token_num(strategy_expanded_x) is " << strategy_expanded_x.at(0)
                  << ", token_num(skip1) is " << strategy_skpi1.at(0) << ", token_num(expanded_row_idx) is "
                  << strategy_rowidx.at(0) << ", token_num(strategy_scales) is " << strategy_scales.at(0)
                  << ", token_num(strategy_expertidx) is " << strategy_expertidx.at(0);
    return FAILED;
  }

  if (strategy_expanded_x.at(0) != 1) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The token num can't be shard, but got"
                  << " shard num : " << strategy_expanded_x.at(0);
    return FAILED;
  }

  // hidden = h
  if (strategy_expanded_x.at(1) != strategy_skpi1.at(1) || strategy_expanded_x.at(1) != strategy_bias.at(1)) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: hidden(expanded_x) == hidden(skip1) == hidden(bias)."
                  << " But current hidden(expanded_x) is " << strategy_expanded_x.at(1) << ", hidden(skip1) is "
                  << strategy_skpi1.at(1) << ", hidden(bias) is " << strategy_rowidx.at(1);
    return FAILED;
  }

  if (strategy_expanded_x.at(1) != 1) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The hidden can't be shard, but got"
                  << " shard num : " << strategy_expanded_x.at(1);
    return FAILED;
  }

  // E = expert_num
  if (strategy_bias.at(0) != 1) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The expert numt can't be shard, but got"
                  << " shard num : " << strategy_expanded_x.at(1);
    return FAILED;
  }

  // k = topk
  if (strategy_scales.at(1) != strategy_expertidx.at(1)) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: k(scales) == k(expertidx)."
                  << " But current k(scales) is " << strategy_scales.at(1) << ", k(expertidx) is "
                  << strategy_expertidx.at(1);
    return FAILED;
  }

  if (strategy_scales.at(1) != 1) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The k can't be shard, but got"
                  << " shard num : " << strategy_scales.at(1);
    return FAILED;
  }

  return SUCCESS;
}

Status MoeFinalizeRoutingInfo::InferDevMatrixShape() {
  auto input_strategys = strategy()->GetInputDim();
  auto strategy_skpi1 = input_strategys.at(kskip1_);  // (N,  h)
  auto strategy_bias = input_strategys.at(kscales_);  // (E,  k)

  // strategy_skpi1  (token_num,  hidden)
  // strategy_bias   (expert_num, k)
  // token_num(N), expert_num(E), hidden(h)
  //      2              1            0
  dev_matrix_shape_ = {strategy_skpi1.at(0), strategy_bias.at(0), strategy_skpi1.at(1)};
  return SUCCESS;
}

Status MoeFinalizeRoutingInfo::InferTensorMap() {
  Shape expanded_x_tensor_map{2, 0};
  Shape skip1_tensor_map{2, 0};
  Shape bias_tensor_map{1, 0};
  Shape scales_tensor_map{2, -1};
  Shape row_idx_row_tensor_map{2};
  Shape expert_idx_tensor_map{2, 0};

  inputs_tensor_map_.emplace_back(expanded_x_tensor_map);
  inputs_tensor_map_.emplace_back(skip1_tensor_map);
  if (input_nums_ == maxInputNum) {
    Shape skip2opt_tensor_map{2, 0};
    inputs_tensor_map_.emplace_back(skip2opt_tensor_map);
  }
  inputs_tensor_map_.emplace_back(bias_tensor_map);
  inputs_tensor_map_.emplace_back(scales_tensor_map);
  inputs_tensor_map_.emplace_back(row_idx_row_tensor_map);
  inputs_tensor_map_.emplace_back(expert_idx_tensor_map);

  Shape output_tensor_map{2, 0};
  outputs_tensor_map_.emplace_back(output_tensor_map);

  return SUCCESS;
}

REGISTER(MoeFinalizeRoutingInfo);
}  // namespace parallel
}  // namespace mindspore
