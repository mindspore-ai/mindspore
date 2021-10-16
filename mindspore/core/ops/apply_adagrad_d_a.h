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

#ifndef MINDSPORE_CORE_OPS_APPLY_ADAGRAD_D_A_H_
#define MINDSPORE_CORE_OPS_APPLY_ADAGRAD_D_A_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "abstract/abstract_value.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameApplyAdagradDA = "ApplyAdagradDA";
/// \brief Update var according to the proximal adagrad scheme.
/// Refer to Python API @ref mindspore.ops.ApplyAdagradDA for more details.
class ApplyAdagradDA : public PrimitiveC {
 public:
  /// \brief Constructor.
  ApplyAdagradDA() : PrimitiveC(kNameApplyAdagradDA) {
    InitIOName({"var", "gradient_accumulator", "gradient_squared_accumulator", "grad", "lr", "l1", "l2", "global_step"},
               {"var", "gradient_accumulator", "gradient_squared_accumulator"});
  }

  /// \brief Destructor.
  ~ApplyAdagradDA() = default;

  MS_DECLARE_PARENT(ApplyAdagradDA, PrimitiveC);
};

AbstractBasePtr ApplyAdagradDAInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args);

using PrimApplyAdagradDAPtr = std::shared_ptr<ApplyAdagradDA>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_APPLY_ADAGRAD_D_A_H_
