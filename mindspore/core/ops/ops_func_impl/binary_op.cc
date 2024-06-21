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

#include "ops/ops_func_impl/binary_op.h"
#include <vector>
#include <memory>
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr BinaryOpFuncImpl::InferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) const {
  const std::string op_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input_number", SizeToLong(input_args.size()), kGreaterEqual, kSize2,
                                           op_name);
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex1]);
  auto x_shape_ptr = input_args[kInputIndex0]->GetShape();
  auto x_shape = x_shape_ptr->GetShapeVector();
  auto y_shape_ptr = input_args[kInputIndex1]->GetShape();
  auto y_shape = y_shape_ptr->GetShapeVector();
  auto output_shape = CalBroadCastShape(x_shape, y_shape, primitive->name());
  return std::make_shared<abstract::TensorShape>(output_shape);
}

TypePtr BinaryOpFuncImpl::InferType(const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const {
  auto input_type = input_args[kInputIndex0]->GetType();
  return input_type->Clone();
}
}  // namespace ops
}  // namespace mindspore
