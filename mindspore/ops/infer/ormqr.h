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

#ifndef MINDSPORE_CORE_OPS_ORMQR_H_
#define MINDSPORE_CORE_OPS_ORMQR_H_
#include <memory>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameOrmqr = "Ormqr";
constexpr auto kAttrLeft = "left";
constexpr auto kAttrTranspose = "transpose";
/// \brief  Computes the matrix-matrix multiplication of Householder matrices with a general matrix.
class OPS_API Ormqr : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Ormqr);
  /// \brief Constructor.
  Ormqr() : BaseOperator(kNameOrmqr) { InitIOName({"x", "tau", "other"}, {"y"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.ormqr for the inputs.
  void Init(const bool left = true, const bool transpose = false);
  /// \brief Set axis.
  void set_left(const bool left);
  /// \brief Set output_type.
  void set_transpose(const bool transpose);
  /// \brief Get left.
  ///
  /// \return left.
  bool get_left() const;
  /// \brief Get transpose.
  ///
  /// \return transpose.
  bool get_transpose() const;
};

OPS_API abstract::AbstractBasePtr OrmqrInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimOrmqrPtr = std::shared_ptr<Ormqr>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_ORMQR_H_
