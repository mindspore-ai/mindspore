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

#ifndef MINDSPORE_CORE_OPS_LOGICAL_XOR_H_
#define MINDSPORE_CORE_OPS_LOGICAL_XOR_H_
#include <memory>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameLogicalXor = "LogicalXor";
/// \brief Computes the truth value of x1 XOR x2, element-wise.
/// Refer to Python API @ref mindspore.numpy.logical_xor for more details.
class MIND_API LogicalXor : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(LogicalXor);
  /// \brief Constructor.
  LogicalXor() : BaseOperator(kNameLogicalXor) { InitIOName({"x", "y"}, {"output"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.numpy.logical_xor for the inputs.
  void Init() const {}
};

MIND_API abstract::AbstractBasePtr LogicalXorInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                   const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimLogicalXorPtr = std::shared_ptr<LogicalXor>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_LOGICAL_XOR_H_
