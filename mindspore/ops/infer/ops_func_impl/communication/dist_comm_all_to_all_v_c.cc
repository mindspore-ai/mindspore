/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/communication/dist_comm_all_to_all_v_c.h"
#include <memory>
#include <string>
#include <functional>
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/ms_context.h"
#include "infer/ops_func_impl/communication/op_comm_func_impl.h"

namespace mindspore {
namespace ops {
ShapeArray DistCommAllToAllVCFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                  const InferInfoPtrList &input_infos) const {
  auto send_count_matrix_opt = input_infos[kIndex3]->GetArrayValue<int64_t>();
  MS_CHECK_VALUE(send_count_matrix_opt.has_value(),
                 primitive->name() + " error: send_count_matrix input should has valid value.");

  const auto &send_count_matrix = send_count_matrix_opt.value();
  auto &value = input_infos[kIndex4];
  auto rank_size = CheckRankSize(primitive->name(), value);
  auto local_rank = GetRankValue(primitive->name(), input_infos[kIndex5]);
  if (rank_size * rank_size != send_count_matrix.size()) {
    MS_LOG(INTERNAL_EXCEPTION) << "The size of the one-dimensional array cannot form a square matrix.";
  }
  int64_t output_numel = 0;
  for (uint64_t i = 0; i < rank_size; ++i) {
    for (uint64_t j = 0; j < rank_size; ++j) {
      if (local_rank == j) {
        output_numel += send_count_matrix[i * rank_size + j];
      }
    }
  }
  if (output_numel == 0) {
    return {ShapeVector{}};
  }
  auto input_shape = input_infos[kIndex0]->GetShape();
  auto numel =
    std::accumulate(input_shape.begin(), input_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
  if (numel != output_numel) {
    MS_LOG(INTERNAL_EXCEPTION) << "The output shape size is not equal to the result from send_count_matrix.";
  }
  return {ShapeVector{output_numel}};
}

std::vector<TypeId> DistCommAllToAllVCFuncImpl::InferType(const PrimitivePtr &primitive,
                                                          const InferInfoPtrList &input_infos) const {
  auto type = input_infos[kIndex1]->GetType();
  auto out_type = input_infos[kIndex0]->GetType();
  return {CheckInferTypes(primitive->name(), type, out_type)};
}
}  // namespace ops
}  // namespace mindspore
