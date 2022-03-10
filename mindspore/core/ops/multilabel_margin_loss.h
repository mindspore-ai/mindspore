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

#ifndef MINDSPORE_CORE_OPS_MULTILABEL_MARGIN_LOSS_H_
#define MINDSPORE_CORE_OPS_MULTILABEL_MARGIN_LOSS_H_
#include <memory>
#include <map>
#include <vector>
#include <set>
#include <string>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
constexpr auto kNameMultilabelMarginLoss = prim::kMultilabelMarginLoss;
/// \brief Creates a criterion that optimizes a multi-class multi-classification hinge loss.
/// Refer to Python API @ref mindspore.ops.MultilabelMarginLoss for more details.
class MS_CORE_API MultilabelMarginLoss : public PrimitiveC {
 public:
  /// \brief Constructor.
  MultilabelMarginLoss() : PrimitiveC(kNameMultilabelMarginLoss) { InitIOName({"x", "target"}, {"y", "is_target"}); }
  /// \brief Destructor.
  ~MultilabelMarginLoss() = default;
  MS_DECLARE_PARENT(MultilabelMarginLoss, PrimitiveC);
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.MultilabelMarginLoss for the inputs.
  void Init() const {}
};

AbstractBasePtr MultilabelMarginLossInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args);
using PrimMultilabelMarginLossPtr = std::shared_ptr<MultilabelMarginLoss>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_MULTILABEL_MARGIN_LOSS_H_
