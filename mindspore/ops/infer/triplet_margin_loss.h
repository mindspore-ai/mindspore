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

#ifndef MINDSPORE_CORE_OPS_TRIPLET_MARGIN_LOSS_H_
#define MINDSPORE_CORE_OPS_TRIPLET_MARGIN_LOSS_H_
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameTripletMarginLoss = "TripletMarginLoss";
class OPS_API TripletMarginLoss : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(TripletMarginLoss);
  TripletMarginLoss() : BaseOperator(kNameTripletMarginLoss) {
    InitIOName({"x", "positive", "negative", "margin"}, {"y"});
  }
  void Init(const int64_t p = 2, const float eps = 1e-6, const bool swap = false,
            const std::string &reduction = "mean");
  void set_p(const int64_t p);
  void set_eps(const float eps);
  void set_swap(const bool swap);
  void set_reduction(const std::string &reduction);
  int64_t get_p() const;
  float get_eps() const;
  bool get_swap() const;
  std::string get_reduction() const;
};

OPS_API abstract::AbstractBasePtr TripletMarginLossInfer(const abstract::AnalysisEnginePtr &,
                                                         const PrimitivePtr &primitive,
                                                         const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimTripletMarginLossPtr = std::shared_ptr<TripletMarginLoss>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_TRIPLET_MARGIN_LOSS_H_
