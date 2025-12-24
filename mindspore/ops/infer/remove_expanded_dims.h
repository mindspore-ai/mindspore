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

#ifndef MINDSPORE_CORE_OPS_REMOVE_EXPANDED_DIMS_H_
#define MINDSPORE_CORE_OPS_REMOVE_EXPANDED_DIMS_H_

#include <tuple>
#include <functional>
#include <vector>
#include "ops/base_operator.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameRemoveExpandedDims = "RemoveExpandedDims";
/// \brief  Removes expanded dimensions in tuple_index and value. Refer to remove_expanded_dims in _compile_utils.py
class OPS_API RemoveExpandedDims : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(RemoveExpandedDims);
  /// \brief Constructor.
  RemoveExpandedDims() : BaseOperator(kNameRemoveExpandedDims) {}
  /// \brief Init function.
  void Init() const {}
  static std::tuple<int64_t, ShapeVector, int64_t> ConstRemoveExpandedDims(
    bool has_true, bool has_false, bool has_sequence, const ShapeVector &broadcast_shape, int64_t rem_ndim,
    const ShapeVector &value_shape, const ShapeVector &data_shape, bool indices_out_empty, int64_t idx_advanced,
    std::vector<int64_t> new_tuple_index_types, size_t expand_index_count);
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_REMOVE_EXPANDED_DIMS_H_
