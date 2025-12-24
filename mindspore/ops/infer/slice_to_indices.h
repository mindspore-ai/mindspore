/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_SLICE_TO_INDICES_H_
#define MINDSPORE_CORE_OPS_SLICE_TO_INDICES_H_

#include <vector>
#include "ops/base_operator.h"
#include "mindapi/base/shape_vector.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSliceToIndices = "SliceToIndices";
/// \brief Normalize Slice index info start, stop, step when data shape is dynamic.
// input: data_shape, init_by_none, start, stop, step
// outputs: index, value_shape, start, stop, step
class OPS_API SliceToIndices : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(SliceToIndices);
  /// \brief Constructor.
  SliceToIndices() : BaseOperator(kNameSliceToIndices) {}
  /// \brief Init function.
  void Init() const {}
};

OPS_API std::vector<int64_t> CalSliceToIndices(const ShapeVector &data_shape, size_t index_axis,
                                               int64_t expand_dims_mask, const std::vector<int64_t> &tuple_index_types,
                                               const std::vector<int64_t> &init_by_none, int64_t *start, int64_t *stop,
                                               int64_t *step);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SLICE_TO_INDICES_H_
