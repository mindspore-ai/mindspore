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

#ifndef MINDSPORE_CORE_OPS_TRIPLET_MARGIN_LOSS_H_
#define MINDSPORE_CORE_OPS_TRIPLET_MARGIN_LOSS_H_
#include <memory>
#include <map>
#include <vector>
#include <set>
#include <string>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameTripletMarginLoss = "TripletMarginLoss";
class TripletMarginLoss : public PrimitiveC {
 public:
  TripletMarginLoss() : PrimitiveC(kNameTripletMarginLoss) {
    InitIOName({"x", "positive", "negative", "margin"}, {"y"});
  }
  ~TripletMarginLoss() = default;
  MS_DECLARE_PARENT(TripletMarginLoss, PrimitiveC);
};

AbstractBasePtr TripletMarginLossInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args);
using PrimTripletMarginLossPtr = std::shared_ptr<TripletMarginLoss>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_TRIPLET_MARGIN_LOSS_H_
