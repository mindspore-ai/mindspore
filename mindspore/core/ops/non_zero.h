/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_NON_ZERO_H_
#define MINDSPORE_CORE_OPS_NON_ZERO_H_

#include <vector>
#include <memory>
#include "ops/base_operator.h"
#include "mindapi/base/types.h"
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"

namespace mindspore {
namespace ops {
constexpr auto kNameNonZero = "NonZero";
/// \brief Calculate tensor not zero index, by default.
/// Refer to Python API @ref mindspore.ops.NonZero for more details.
class MIND_API NonZero : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(NonZero);
  /// \brief Constructor.
  NonZero() : BaseOperator(kNameNonZero) { InitIOName({"x"}, {"output"}); }
};
AbstractBasePtr NonZeroInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args);
using PrimNonZeroPtr = std::shared_ptr<NonZero>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_NON_ZERO_H_
