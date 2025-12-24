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
 * limitations under the License.
 */

#ifndef MINDSPORE_CORE_OPS_ARG_MIN_V2_H_
#define MINDSPORE_CORE_OPS_ARG_MIN_V2_H_
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameArgminV2 = "ArgminV2";
/// \brief Returns the indices of the minimum value of a tensor across the axis.
/// Refer to Python API @ref mindspore.ops.ArgminV2 for more details.
class OPS_API ArgminV2 : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ArgminV2);
  /// \brief Constructor.
  ArgminV2() : BaseOperator(kNameArgminV2) { InitIOName({"x", "axis"}, {"y"}); }
  explicit ArgminV2(const std::string k_name) : BaseOperator(k_name) { InitIOName({"x", "axis"}, {"y"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.ArgminV2 for the inputs.
  void Init();
};
OPS_API abstract::AbstractBasePtr ArgminV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimArgminV2 = std::shared_ptr<ArgminV2>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_ARG_MIN_V2_H_
