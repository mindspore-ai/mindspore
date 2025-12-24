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

#ifndef MINDSPORE_CORE_OPS_AVG_POOL_GRAD_V1_H_
#define MINDSPORE_CORE_OPS_AVG_POOL_GRAD_V1_H_
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameAvgPoolGradV1 = "AvgPoolGradV1";
class OPS_API AvgPoolGradV1 : public BaseOperator {
 public:
  AvgPoolGradV1() : BaseOperator(kNameAvgPoolGradV1) { InitIOName({"orig_input_shape", "grad"}, {"output"}); }
  MIND_API_BASE_MEMBER(AvgPoolGradV1);
};

OPS_API abstract::AbstractBasePtr AvgPoolGradV1Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                     const std::vector<AbstractBasePtr> &input_args);
using PrimAvgPoolGradV1Ptr = std::shared_ptr<AvgPoolGradV1>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_AVG_POOL_GRAD_V1_H_
