/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_LP_NORM_UPDATE_V2_H_
#define MINDSPORE_CORE_OPS_LP_NORM_UPDATE_V2_H_
#include <memory>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameLpNormUpdateV2 = "LpNormUpdateV2";
class MIND_API LpNormUpdateV2 : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(LpNormUpdateV2);
  LpNormUpdateV2() : BaseOperator(kNameLpNormUpdateV2) { InitIOName({"input"}, {"output"}); }

  void Init(const float p = 2.0, const float epsilon = 1e-12);

  void set_p(const float p);

  void set_epsilon(const float epsilon);

  float get_p() const;

  float get_epsilon() const;
};

MIND_API abstract::AbstractBasePtr LpNormUpdateV2Infer(const abstract::AnalysisEnginePtr &,
                                                       const PrimitivePtr &primitive,
                                                       const std::vector<abstract::AbstractBasePtr> &input_args);

using PrimLpNormUpdateV2Ptr = std::shared_ptr<LpNormUpdateV2>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_LP_NORM_UPDATE_V2_H_
