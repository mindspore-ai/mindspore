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

#ifndef MINDSPORE_CORE_OPS_GET_TUPLE_INDEX_INFO_H_
#define MINDSPORE_CORE_OPS_GET_TUPLE_INDEX_INFO_H_

#include <vector>
#include <string>
#include "ops/base_operator.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameGetTupleIndexInfo = "GetTupleIndexInfo";
/// \brief Get tuple index info. Refer to generate_index_info_from_tuple_of_mixed_tensors in _constexpr_utils.py.
// inputs: data, fancy_index_position, tensor0, tensor1 ..., tensor7
// attrs: kTupleIndexTypes
// outputs: broadcast_shape, final_shape, index_tensor_new_shape, fancy_index_position,
// slice_shape0 ..., slice_shape7
class OPS_API GetTupleIndexInfo : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(GetTupleIndexInfo);
  /// \brief Constructor.
  GetTupleIndexInfo() : BaseOperator(kNameGetTupleIndexInfo) {}
  /// \brief Init function.
  void Init() const {}
  static std::vector<ShapeVector> ConstGetTupleIndexInfo(const ShapeVector &data_shape,
                                                         const std::vector<ShapeVector> &tensor_shapes,
                                                         const std::vector<int64_t> &tuple_index_types,
                                                         ShapeVector *broadcast_shape, ShapeVector *final_shape,
                                                         ShapeVector *index_tensor_new_shape, size_t *fancy_position,
                                                         const string &tuple_index_info_type);
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_GET_TUPLE_INDEX_INFO_H_
