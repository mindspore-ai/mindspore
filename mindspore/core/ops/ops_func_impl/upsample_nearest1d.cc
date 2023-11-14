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

#include "ops/ops_func_impl/upsample_nearest1d.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr UpsampleNearest1dFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  if (input_args.size() != 3) {
    MS_LOG(EXCEPTION) << "input args size should be 3, but got " << input_args.size();
  }
  const auto &prim_name = primitive->name();
  auto input_shape_ptr = input_args[kInputIndex0]->GetShape();
  MS_EXCEPTION_IF_NULL(input_shape_ptr);
  const auto &input_shape = input_shape_ptr->GetShapeVector();
  auto output_size_value = input_args[1]->GetValue();
  MS_EXCEPTION_IF_NULL(output_size_value);
  std::vector<int64_t> output_size_shape = GetValue<std::vector<int64_t>>(output_size_value);

  if (input_shape.size() != 3) {
    MS_LOG(EXCEPTION) << "For '" << prim_name << "', input tensor must be a 3D Tensor, but got:" << input_shape.size();
  }

  ShapeVector ret_shape{input_shape[0], input_shape[1], output_size_shape[0]};
  return std::make_shared<abstract::Shape>(ret_shape);
}

TypePtr UpsampleNearest1dFuncImpl::InferType(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  const auto &prim_name = primitive->name();
  auto input_type = input_args[kInputIndex0]->GetType();
  MS_EXCEPTION_IF_NULL(input_type);

  const std::set<TypePtr> &valid_types = {kFloat32, kFloat16, kUInt8};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input", input_type, valid_types, prim_name);
  return input_type->Clone();
}
}  // namespace ops
}  // namespace mindspore
