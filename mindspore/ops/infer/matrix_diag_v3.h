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

#ifndef MINDSPORE_CORE_OPS_MATRIX_DIAG_V3_H_
#define MINDSPORE_CORE_OPS_MATRIX_DIAG_V3_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameMatrixDiagV3 = "MatrixDiagV3";

/// \brief Returns a batched diagonal tensor with given batched diagonal values.
/// Refer to Python API @ref mindspore.ops.MatrixDiagV3 for more details.
class OPS_API MatrixDiagV3 : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(MatrixDiagV3);
  /// \brief Constructor.
  MatrixDiagV3() : BaseOperator(kNameMatrixDiagV3) {
    InitIOName({"x", "k", "num_rows", "num_cols", "padding_value"}, {"y"});
  }

  void Init(const std::string &align = "RIGHT_LEFT");

  void set_align(const std::string &align);

  std::string get_align() const;
};

OPS_API abstract::AbstractBasePtr MatrixDiagV3Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                    const std::vector<abstract::AbstractBasePtr> &input_args);

using PrimMatrixDiagV3Ptr = std::shared_ptr<MatrixDiagV3>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_MATRIX_DIAG_V3_H_
