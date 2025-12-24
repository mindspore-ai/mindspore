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

#ifndef MINDSPORE_CORE_OPS_NORMALIZE_DIM_INDEX_H_
#define MINDSPORE_CORE_OPS_NORMALIZE_DIM_INDEX_H_

#include <vector>
#include "ops/base_operator.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameNormalizeDimIndex = "NormalizeDimIndex";
/// \brief Normalize index axis in tuple indices.
// data[a, b, ... c]
// data_shape [2,2,2,2,2]
// index axis of index 'c' is 4th
// input: data, dim_index, expand_dims_cnt(used in getitem by tuple)
// attr: tuple_index_types
// outputs: normalized_dim_index
class OPS_API NormalizeDimIndex : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(NormalizeDimIndex);
  /// \brief Constructor.
  NormalizeDimIndex() : BaseOperator(kNameNormalizeDimIndex) {}
  /// \brief Init function.
  void Init() const {}
  static size_t ConstNormalizeDimIndex(size_t data_dims, size_t dim_index,
                                       const std::vector<int64_t> &tuple_index_types, size_t expand_dims_mask);
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_NORMALIZE_DIM_INDEX_H_
