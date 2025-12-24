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

#ifndef MINDSPORE_CORE_OPS_MULNONAN_H_
#define MINDSPORE_CORE_OPS_MULNONAN_H_
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameMulNoNan = "MulNoNan";
class OPS_API MulNoNan : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(MulNoNan);
  MulNoNan() : BaseOperator(kNameMulNoNan) { InitIOName({"x", "y"}, {"output"}); }
  explicit MulNoNan(const std::string k_name) : BaseOperator(k_name) { InitIOName({"x", "y"}, {"output"}); }
  void Init() const {}
};
OPS_API abstract::AbstractBasePtr MulNoNanInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                const std::vector<abstract::AbstractBasePtr> &input_args);
using kPrimMulNoNanPtr = std::shared_ptr<MulNoNan>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_MULNONAN_H_
