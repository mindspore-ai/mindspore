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

#ifndef MINDSPORE_CORE_OPS_SSPADDMM_H_
#define MINDSPORE_CORE_OPS_SSPADDMM_H_
#include <memory>
#include <set>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSspaddmm = "Sspaddmm";
/// \brief Performs a matrix multiplication of the matrices mat1 and mat2.
/// The matrix input is added to the final result.
/// Refer to Python API @ref mindspore.ops.Sspaddmm for more details.
class OPS_API Sspaddmm : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Sspaddmm);
  /// \brief Constructor.
  Sspaddmm() : BaseOperator(kNameSspaddmm) {
    InitIOName(
      {"x1_indices", "x1_values", "x1_shape", "x2_indices", "x2_values", "x2_shape", "x3_dense", "alpha", "beta"},
      {"y_indices", "y_values", "y_shape"});
  }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.Sspaddmm for the inputs.
  void Init() const {}
};
OPS_API abstract::AbstractBasePtr SspaddmmInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SSPADDMM_H_
