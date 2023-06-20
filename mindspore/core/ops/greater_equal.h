/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_GREATER_EQUAL_H_
#define MINDSPORE_CORE_OPS_GREATER_EQUAL_H_
#include <memory>
#include <vector>

#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameGreaterEqual = "GreaterEqual";
/// \brief Computes the boolean value of \f$x>=y\f$ element-wise.
/// Refer to Python API @ref mindspore.ops.GreaterEqual for more details.
class MIND_API GreaterEqual : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(GreaterEqual);
  /// \brief Constructor.
  GreaterEqual() : BaseOperator(kNameGreaterEqual) {}
};
MIND_API abstract::AbstractBasePtr GreaterEqualInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                     const std::vector<abstract::AbstractBasePtr> &input_args);
using kPrimGreaterEqualPtr = std::shared_ptr<GreaterEqual>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_GREATER_EQUAL_H_
