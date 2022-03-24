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

#ifndef MINDSPORE_CORE_OPS_POW_H_
#define MINDSPORE_CORE_OPS_POW_H_
#include <map>
#include <vector>
#include <string>
#include <memory>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNamePow = "Pow";
/// \brief Computes a tensor to the power of the second input.
/// Refer to Python API @ref mindspore.ops.Pow for more details.
class MIND_API Pow : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Pow);
  /// \brief Constructor.
  explicit Pow(const std::string &k_name = kNamePow) : BaseOperator(k_name) { InitIOName({"x", "y"}, {"output"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.Pow for the inputs.
  void Init();
};
abstract::AbstractBasePtr PowInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<abstract::AbstractBasePtr> &input_args);
using kPrimPowPtr = std::shared_ptr<Pow>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_POW_H_
