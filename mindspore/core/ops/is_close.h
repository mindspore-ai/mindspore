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

#ifndef MINDSPORE_CORE_OPS_IsClose_H_
#define MINDSPORE_CORE_OPS_IsClose_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameIsClose = "IsClose";
class MIND_API IsClose : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(IsClose);
  IsClose() : BaseOperator(kNameIsClose) { InitIOName({"x1", "x2"}, {"y"}); }

  void Init(const float rtol, const float atol, const bool equal_nan);

  void set_rtol(const float rtol);

  void set_atol(const float atol);

  void set_equal_nan(const bool equal_nan);

  float get_rtol() const;

  float get_atol() const;

  bool get_equal_nan() const;
};
MIND_API abstract::AbstractBasePtr IsCloseInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimIsClosePtr = std::shared_ptr<IsClose>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_IsClose_H_
