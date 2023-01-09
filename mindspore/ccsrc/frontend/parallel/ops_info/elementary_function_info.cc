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

#include "frontend/parallel/ops_info/elementary_function_info.h"

#include "frontend/parallel/dynamic_creator.h"

namespace mindspore {
namespace parallel {
Status CholeskyInfo::GetAttrs() {
  axis_ = {-2, -1};
  return SUCCESS;
}

// the last two dimensions can not be split
Status CholeskyInfo::CheckStrategy(const mindspore::parallel::StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }

  Strategies stra = strategy->GetInputDim();
  Dimensions input_strategy = stra.at(0);

  for (auto &element : axis_) {
    int64_t axis_index = element;
    if (element < 0) {
      size_t input_dim = inputs_shape_.at(0).size();
      axis_index = static_cast<int64_t>(input_dim) + element;
    }

    int64_t axis_strategy = input_strategy.at(LongToSize(axis_index));
    // Dimension corresponding to axis is un-splittable
    if (axis_strategy != MIN_SLICE_NUM) {
      MS_LOG(ERROR) << name_ << ": The last two dimensions can not be split, but the strategy is " << input_strategy;
      return FAILED;
    }
  }

  return SUCCESS;
}

REGISTER(ExpInfo);
REGISTER(LogInfo);
REGISTER(CosInfo);
REGISTER(ACosInfo);
REGISTER(LogicalNotInfo);
REGISTER(AbsInfo);
REGISTER(SignInfo);
REGISTER(FloorInfo);
REGISTER(RoundInfo);
REGISTER(ReciprocalInfo);
REGISTER(InvInfo);
REGISTER(RsqrtInfo);
REGISTER(TanInfo);
REGISTER(SinInfo);
REGISTER(SinhInfo);
REGISTER(Log1pInfo);
REGISTER(Expm1Info);
REGISTER(CoshInfo);
REGISTER(CeilInfo);
REGISTER(CholeskyInfo);
REGISTER(AtanhInfo);
REGISTER(AtanInfo);
REGISTER(AsinInfo);
REGISTER(AsinhInfo);
REGISTER(AcoshInfo);
REGISTER(ErfInfo);
REGISTER(ErfcInfo);
REGISTER(ZerosLikeInfo);
REGISTER(OnesLikeInfo);
REGISTER(BesselI0eInfo);
REGISTER(BesselI1eInfo);
REGISTER(BesselI0Info);
REGISTER(BesselI1Info);
REGISTER(BesselJ0Info);
REGISTER(BesselJ1Info);
REGISTER(LgammaInfo);
REGISTER(TruncInfo);
}  // namespace parallel
}  // namespace mindspore
