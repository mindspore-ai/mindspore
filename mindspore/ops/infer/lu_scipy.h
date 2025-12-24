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
 * limitations under the License
 */

#ifndef MINDSPORE_CORE_OPS_LU_SCIPY_H_
#define MINDSPORE_CORE_OPS_LU_SCIPY_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameLU = "LU";
class OPS_API LU : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(LU);
  LU() : BaseOperator(kNameLU) { InitIOName({"x"}, {"lu", "pivots", "permutation"}); }
};
abstract::AbstractBasePtr LUInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimLUPtr = std::shared_ptr<LU>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_LU_SCIPY_H_
