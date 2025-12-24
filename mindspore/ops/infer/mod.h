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

#ifndef MINDSPORE_CORE_OPS_MOD_H_
#define MINDSPORE_CORE_OPS_MOD_H_
#include <memory>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameMod = "Mod";
/// \brief Computes the remainder of dividing the first input tensor by the second input tensor element-wise.
/// Refer to Python API @ref mindspore.ops.Mod for more details.
class OPS_API Mod : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Mod);
  /// \brief Constructor.
  Mod() : BaseOperator(kNameMod) { InitIOName({"x", "y"}, {"output"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.Mod for the inputs.
  void Init() const {}
};
OPS_API OPS_API abstract::AbstractBasePtr ModInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                   const std::vector<abstract::AbstractBasePtr> &input_args);
using kPrimModPtr = std::shared_ptr<Mod>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_MOD_H_
