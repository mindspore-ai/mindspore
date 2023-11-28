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

#include <utility>
#include <algorithm>

#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/ops_info/tril_info.h"

namespace mindspore {
namespace parallel {
Status TrilInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }

  auto input_num = strategy->GetInputNumber();
  if (input_num != 1) {
    MS_LOG(ERROR) << name_ << ": The strategy size must be equal to 1"
                  << ", but got strategie size is " << input_num;
    return FAILED;
  }
  // The strategy for each input tensor must be equal
  Strategies strategies = strategy->GetInputDim();
  auto stra_value = strategies.at(0);
  constexpr size_t smallest_stra_len = 2;
  if (stra_value.size() < smallest_stra_len) {
    MS_LOG(ERROR) << name_ << ": The strategy value size must be greater than 2"
                  << ", but got strategie value size is " << stra_value.size();
    return FAILED;
  }
  return SUCCESS;
}

std::vector<StrategyPtr> TrilInfo::GenerateOpStrategies(int64_t stage_id) {
  if ((inputs_shape_.size() != 1)) {
    MS_LOG(EXCEPTION) << name_ << " : Inputs shape size is wrong.";
  }

  Shape input0_split(inputs_shape_.at(0).size(), 1);
  Shapes splittable_inputs = {input0_split};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << " : Generate strategies for independent inputs failed.";
  }
  return sp_vector;
}

int64_t TrilInfo::GetDiag() {
  const auto &input_shape = inputs_shape_.at(0);
  auto row = *(input_shape.rbegin() + 1);
  auto col = *(input_shape.rbegin());
  auto stra = strategy();
  Strategies strategies = stra->GetInputDim();
  auto stra_value = strategies.at(0);
  auto c = *(stra_value.rbegin() + 1);
  auto d = *(stra_value.rbegin());
  int64_t rank = g_device_manager->rank_index_in_stage();
  auto t = row / c;
  auto u = col / d;
  // represent position in the row
  int64_t m = 0;
  // represent position in the col
  int64_t n = 0;
  if (repeated_calc_num_ > 1) {
    // repeated calc
    auto h = *(dev_matrix_shape().rbegin());
    m = (rank / d / h % c) * t;
    n = (rank / h % d) * u;
  } else {
    m = (rank / d % c) * t;
    n = (rank % d) * u;
  }
  // numbers to be retained in the first complete row.
  auto x = m + diagonal_ + 1;
  // numbers removed from the left of the first row
  auto y = n;
  // Numbers to be reserved in the first row of the fragment.
  auto z = x - y;
  // clip
  if (z > u) {
    z = u;
  } else if (z < -t + 1) {
    z = -t + 1;
  }
  return z - 1;
}

void TrilInfo::ReplaceNodeInputOrAttrs() {
  for (auto &node : cnodes_) {
    auto prim = GetValueNode<PrimitivePtr>(node->input(0));
    auto new_diag = GetDiag();
    prim->set_attr("diagonal", MakeValue(new_diag));
  }
}

Status TrilInfo::GetAttrs() {
  diagonal_ = GetIntAttr("diagonal");
  return SUCCESS;
}

REGISTER(TrilInfo);
}  // namespace parallel
}  // namespace mindspore
