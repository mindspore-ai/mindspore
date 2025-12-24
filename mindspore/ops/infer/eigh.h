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

#ifndef MINDSPORE_CORE_OPS_EIGH_H_
#define MINDSPORE_CORE_OPS_EIGH_H_

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
/// \brief Computes the eigenvalue decomposition of a (batched) square matrix.
/// Refer to Python API @ref mindspore.ops.Eig for more details.
constexpr auto kNameEigh = "Eigh";
class OPS_API Eigh : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Eigh);
  /// \brief Constructor.
  Eigh() : BaseOperator(kNameEigh) { InitIOName({"A"}, {"output_w", "output_v"}); }
  bool get_compute_eigen_vectors() const;
  bool get_lower() const;
};
OPS_API abstract::AbstractBasePtr EighInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_EIGH_H_
