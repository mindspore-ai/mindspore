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

#ifndef MINDSPORE_CORE_OPS_TENSOR_MOVE_ELEMENTS_H_
#define MINDSPORE_CORE_OPS_TENSOR_MOVE_ELEMENTS_H_
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameTensorMove = "TensorMove";
/// \brief Updates tensor values by using input indices and value.
/// Refer to Python API @ref mindspore.ops.TensorMove for more details.
class OPS_API TensorMove : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(TensorMove);
  /// \brief Constructor.
  TensorMove() : BaseOperator(kNameTensorMove) { InitIOName({"input"}, {"output"}); }
};
OPS_API abstract::AbstractBasePtr TensorMoveInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                  const std::vector<abstract::AbstractBasePtr> &input_args);
using kPrimTensorMovePtr = std::shared_ptr<TensorMove>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_TENSOR_SCATTER_ELEMENTS_H_
