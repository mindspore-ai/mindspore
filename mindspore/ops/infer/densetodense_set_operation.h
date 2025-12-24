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

#ifndef MINDSPORE_CORE_OPS_DENSE_TO_DENSE_SET_OPERATION_H_
#define MINDSPORE_CORE_OPS_DENSE_TO_DENSE_SET_OPERATION_H_
#include <memory>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameDenseToDenseSetOperation = "DenseToDenseSetOperation";
/// \brief Applies set operation along last dimension of 2 `Tensor` inputs.
/// Refer to Python API @ref mindspore.ops.DenseToDenseSetOperation for more details.
class OPS_API DenseToDenseSetOperation : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(DenseToDenseSetOperation);
  /// \brief Constructor.
  DenseToDenseSetOperation() : BaseOperator(kNameDenseToDenseSetOperation) {
    InitIOName({"x1", "x2"}, {"y_indices", "y_values", "y_shape"});
  }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.DenseToDenseSetOperation for more details.
  void Init() const {}
};
OPS_API abstract::AbstractBasePtr DenseToDenseSetOperationInfer(
  const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
  const std::vector<abstract::AbstractBasePtr> &input_args);
using kDenseToDenseSetOperationPtr = std::shared_ptr<DenseToDenseSetOperation>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_DENSE_TO_DENSE_SET_OPERATION_H_
