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

#include "infer/ops_func_impl/dynamic_block_quant.h"
#include <set>
#include <string>
#include <cmath>
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
ShapeArray DynamicBlockQuantFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                 const InferInfoPtrList &input_infos) const {
  const auto &x_shape = input_infos[kIndex0]->GetShape();
  if (x_shape.size() != kDim2) {
    MS_EXCEPTION(ValueError) << "For DynamicBlockQuant, input x must be 2D tensor, but got " << x_shape.size() << "D.";
  }

  auto M = x_shape[kDim0];
  auto N = x_shape[kDim1];

  auto row_block_size = input_infos[kIndex4]->GetScalarValue<int64_t>().value();
  auto col_block_size = input_infos[kIndex5]->GetScalarValue<int64_t>().value();
  auto scale_row_size = static_cast<int64_t>(std::ceil(M / static_cast<float>(row_block_size)));
  auto scale_col_size = static_cast<int64_t>(std::ceil(N / static_cast<float>(col_block_size)));
  ShapeVector scale_shape = {scale_row_size, scale_col_size};

  ShapeArray shapes_list;
  (void)shapes_list.emplace_back(x_shape);
  (void)shapes_list.emplace_back(std::move(scale_shape));

  return shapes_list;
}

std::vector<TypeId> DynamicBlockQuantFuncImpl::InferType(const PrimitivePtr &primitive,
                                                         const InferInfoPtrList &input_infos) const {
  auto dst_typeid = static_cast<TypeId>(input_infos[kIndex3]->GetScalarValue<int64_t>().value());
  return {dst_typeid, kNumberTypeFloat32};
}
}  // namespace ops
}  // namespace mindspore
