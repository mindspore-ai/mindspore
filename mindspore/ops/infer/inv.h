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

#ifndef MINDSPORE_CORE_OPS_INV_H_
#define MINDSPORE_CORE_OPS_INV_H_
#include <memory>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameInv = "Inv";
class OPS_API Inv : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Inv);
  Inv() : BaseOperator(kNameInv) { InitIOName({"x"}, {"y"}); }
  void Init() const {}
};
OPS_API abstract::AbstractBasePtr InvInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                           const std::vector<abstract::AbstractBasePtr> &input_args);
using kPrimInvPtr = std::shared_ptr<Inv>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_INV_H_
