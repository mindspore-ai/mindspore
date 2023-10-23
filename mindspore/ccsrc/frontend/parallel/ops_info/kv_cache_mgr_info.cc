/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/ops_info/kv_cache_mgr_info.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"

namespace mindspore {
namespace parallel {

Status KVCacheMgrInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

Status KVCacheMgrInfo::GetAttrs() { return SUCCESS; }

Status KVCacheMgrInfo::InferAsLossDivisor() {
  if (outputs_tensor_map_.empty()) {
    MS_LOG(ERROR) << name_ << ": The size of outputs tensor map is empty.";
    return FAILED;
  }
  return SUCCESS;
}

Status KVCacheMgrInfo::CheckStrategy(const StrategyPtr &strategy) { return SUCCESS; }

Status KVCacheMgrInfo::InferDevMatrixShape() {
  Strategies stra = strategy_->GetInputDim();
  Dimensions past_stgy_dim = stra.at(0);
  size_t dp, mp;
  dp = past_stgy_dim[0];
  mp = past_stgy_dim[1];
  dev_matrix_shape_.push_back(dp);
  dev_matrix_shape_.push_back(mp);
  dev_matrix_shape_.push_back(1);
  dev_matrix_shape_.push_back(1);
  return SUCCESS;
}

Status KVCacheMgrInfo::InferTensorMap() {
  inputs_tensor_map_.push_back({3, 2, 1, 0});
  inputs_tensor_map_.push_back({3, 2, 1, 0});
  inputs_tensor_map_.push_back({-1});
  outputs_tensor_map_.push_back({3, 2, 1, 0});
  return SUCCESS;
}

std::vector<StrategyPtr> KVCacheMgrInfo::GenerateOpStrategies(int64_t stage_id) {
  std::vector<StrategyPtr> sp;
  return sp;
}

REGISTER(KVCacheMgrInfo);
}  // namespace parallel
}  // namespace mindspore
