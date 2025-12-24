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

#ifndef MINDSPORE_CORE_OPS_MATRIX_SOLVE_LS_H
#define MINDSPORE_CORE_OPS_MATRIX_SOLVE_LS_H
#include <memory>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameMatrixSolveLs = "MatrixSolveLs";

class OPS_API MatrixSolveLs : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(MatrixSolveLs);

  /// \brief Constructor.
  MatrixSolveLs() : BaseOperator(kNameMatrixSolveLs) { InitIOName({"matrix", "rhs", "l2"}, {"y"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.MatrixSolveLs for the inputs.
  void Init() const {}
};

OPS_API abstract::AbstractBasePtr MatrixSolveLsInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                     const std::vector<abstract::AbstractBasePtr> &input_args);

using PrimMatrixSolveLsPtr = std::shared_ptr<MatrixSolveLs>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_MATRIX_SOLVE_LS_H
