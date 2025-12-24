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

#ifndef MINDSPORE_CORE_OPS_DENSE_TO_SPARSE_SET_OPERATION_H_
#define MINDSPORE_CORE_OPS_DENSE_TO_SPARSE_SET_OPERATION_H_
#include <memory>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameDenseToSparseSetOperation = "DenseToSparseSetOperation";
/// \brief Applies set operation along last dimension of `Tensor` and `SparseTensor`.
class OPS_API DenseToSparseSetOperation : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(DenseToSparseSetOperation);
  DenseToSparseSetOperation() : BaseOperator(kNameDenseToSparseSetOperation) {
    InitIOName({"x1", "x2_indices", "x2_values", "x2_shape"}, {"y_indices", "y_values", "y_shape"});
  }
};
OPS_API abstract::AbstractBasePtr DenseToSparseSetOperationInfer(
  const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
  const std::vector<abstract::AbstractBasePtr> &input_args);
using kDenseToSparseSetOperationPtr = std::shared_ptr<DenseToSparseSetOperation>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_DENSE_TO_SPARSE_SET_OPERATION_H_
