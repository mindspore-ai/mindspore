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

#include "infer/ops_func_impl/convolution_grad.h"

#include <string>
#include <set>
#include <utility>

#include "ops_utils/op_constants.h"
#include "mindapi/base/shape_vector.h"
#include "utils/check_convert_utils.h"
#include "primitive/op_name.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
ShapeArray ConvolutionGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                               const InferInfoPtrList &input_infos) const {
  const auto &dout_info = input_infos[kIndex0];
  const auto &dout_shape = dout_info->GetShape();
  auto x_shape = input_infos[kIndex1]->GetShape();
  auto weight_shape = input_infos[kIndex2]->GetShape();
  auto out_channels = dout_info->IsDynamicRank() ? abstract::Shape::kShapeDimAny : dout_shape.at(kIndex1);
  ShapeVector bias_shape{out_channels};
  ShapeArray output_shapes{std::move(x_shape), std::move(weight_shape), std::move(bias_shape)};
  return output_shapes;
}

std::vector<TypeId> ConvolutionGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                                       const InferInfoPtrList &input_infos) const {
  auto x_type_ptr = input_infos[kInputIndex1]->GetType();
  auto weight_type_ptr = input_infos[kInputIndex2]->GetType();
  return {x_type_ptr, weight_type_ptr, x_type_ptr};
}
}  // namespace ops
}  // namespace mindspore
