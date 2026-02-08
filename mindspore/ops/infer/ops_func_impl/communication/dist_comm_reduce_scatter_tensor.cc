/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/communication/dist_comm_reduce_scatter_tensor.h"
#include <memory>
#include <set>
#include <string>
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/ms_context.h"
#include "infer/ops_func_impl/communication/op_comm_func_impl.h"

namespace mindspore {
namespace ops {
ShapeArray DistCommReduceScatterTensorFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                           const InferInfoPtrList &input_infos) const {
  auto &value = input_infos[kIndex2];
  auto rank_size = CheckRankSize(primitive->name(), value);

  auto input_shape = input_infos[kIndex1]->GetShape();
  if (input_shape.size() == 0) {
    MS_EXCEPTION(ValueError) << "the shape size of 'input_shape' can not be 0.";
  }
  if (input_shape[0] % rank_size != 0) {
    MS_EXCEPTION(ValueError)
      << "the first dimension for 'input_shape' must be divided by 'rank_size', but got input_shape[0]: "
      << input_shape[0] << ", rank_size: " << rank_size;
  }
  input_shape[0] = static_cast<int64_t>(input_shape[0] / rank_size);

  const auto &output_shape = input_infos[kIndex0]->GetShape();
  CheckInferShape(primitive->name(), input_shape, output_shape);
  return {input_shape};
}

std::vector<TypeId> DistCommReduceScatterTensorFuncImpl::InferType(const PrimitivePtr &primitive,
                                                                   const InferInfoPtrList &input_infos) const {
  auto type = input_infos[kIndex1]->GetType();
  auto out_type = input_infos[kIndex0]->GetType();
  return {CheckInferTypes(primitive->name(), type, out_type, true)};
}
}  // namespace ops
}  // namespace mindspore
