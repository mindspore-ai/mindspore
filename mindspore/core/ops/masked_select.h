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

#ifndef MINDSPORE_CORE_OPS_MASKED_SELECT_H_
#define MINDSPORE_CORE_OPS_MASKED_SELECT_H_

#include <memory>
#include <vector>
#include "abstract/abstract_value.h"
#include "mindapi/base/types.h"
#include "ops/base_operator.h"
#include "ops/primitive_c.h"

namespace mindspore {
namespace ops {
constexpr auto kNameMaskedSelect = "MaskedSelect";
/// \brief Returns a new 1-D Tensor which indexes the input tensor according to the boolean mask.
/// Refer to Python API @ref mindspore.ops.MaskedSelect for more details.
class MIND_API MaskedSelect : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(MaskedSelect);
  /// \brief Constructor.
  MaskedSelect() : BaseOperator(kNameMaskedSelect) { InitIOName({"x", "mask"}, {"output"}); }
};
AbstractBasePtr MaskedSelectInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args);
using PrimMaskedSelectPtr = std::shared_ptr<MaskedSelect>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_MASKED_SELECT_H_
