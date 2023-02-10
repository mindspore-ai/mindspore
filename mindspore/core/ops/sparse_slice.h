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

#ifndef MINDSPORE_CORE_OPS_SPARSE_SLICE_H_
#define MINDSPORE_CORE_OPS_SPARSE_SLICE_H_
#include <vector>
#include <set>
#include <memory>
#include "ops/base_operator.h"
#include "mindapi/base/types.h"
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSparseSlice = "SparseSlice";
/// \brief Slices a SparseTensor based on the "start" and "size".
/// Refer to Python API @ref mindspore.ops.SparseSlice for more details.
class MIND_API SparseSlice : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(SparseSlice);
  /// \brief Constructor.
  SparseSlice() : BaseOperator(kNameSparseSlice) {
    InitIOName({"indices", "values", "shape", "start", "size"}, {"y_indices", "y_values", "y_shape"});
  }
  const void Init() const {}
};
AbstractBasePtr SparseSliceInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const std::vector<AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SPARSE_SLICE_H_
