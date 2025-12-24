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

#ifndef MINDSPORE_CORE_OPS_MATRIX_SET_DIAG_V3_H_
#define MINDSPORE_CORE_OPS_MATRIX_SET_DIAG_V3_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameMatrixSetDiagV3 = "MatrixSetDiagV3";

/// \brief Returns a batched matrix tensor with new batched diagonal values.
/// Refer to Python API @ref mindspore.ops.MatrixSetDiagV3 for more details.
class OPS_API MatrixSetDiagV3 : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(MatrixSetDiagV3);
  /// \brief Constructor.
  MatrixSetDiagV3() : BaseOperator(kNameMatrixSetDiagV3) { InitIOName({"x", "diagonal", "k"}, {"y"}); }

  void Init(const std::string &align = "RIGHT_LEFT");

  void set_align(const std::string &align);

  std::string get_align() const;
};

OPS_API abstract::AbstractBasePtr MatrixSetDiagV3Infer(const abstract::AnalysisEnginePtr &,
                                                       const PrimitivePtr &primitive,
                                                       const std::vector<abstract::AbstractBasePtr> &input_args);

using PrimMatrixSetDiagV3Ptr = std::shared_ptr<MatrixSetDiagV3>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_MATRIX_SET_DIAG_V3_H_
