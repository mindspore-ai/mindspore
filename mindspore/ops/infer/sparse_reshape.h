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

#ifndef MINDSPORE_CORE_OPS_SPARSE_RESHAPE_H_
#define MINDSPORE_CORE_OPS_SPARSE_RESHAPE_H_
#include <memory>
#include <vector>

#include "abstract/abstract_value.h"
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSparseReshape = "SparseReshape";
/// \brief Reshapes a SparseTensor to represent values in a new dense shape.
/// Refer to Python API @ref mindspore.ops.SparseReshape for more details.
class OPS_API SparseReshape : public BaseOperator {
 public:
  /// \brief Constructor.
  SparseReshape() : BaseOperator(kNameSparseReshape) {
    InitIOName({"indices", "shape", "new_shape"}, {"y_indices", "y_shape"});
  }
  MIND_API_BASE_MEMBER(SparseReshape);
};
AbstractBasePtr SparseReshapeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_SPARSE_RESHAPE_H_
