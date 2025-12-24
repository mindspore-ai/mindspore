/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#ifndef MINDSPORE_CORE_OPS_EPS_H
#define MINDSPORE_CORE_OPS_EPS_H
#include <memory>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"
namespace mindspore {
namespace ops {
constexpr auto kNameEps = "Eps";
/// \brief Creates a new tensor. The values of all elements are minimum values of the data type.
/// Refer to Python API @ref mindspore.ops.OnesLike for more details.
class OPS_API Eps : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Eps);
  /// \brief Constructor.
  Eps() : BaseOperator(kNameEps) { InitIOName({"x"}, {"y"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.OnesLike for the inputs.
  void Init() const {}
};

abstract::AbstractBasePtr EpsInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_EPS_H
