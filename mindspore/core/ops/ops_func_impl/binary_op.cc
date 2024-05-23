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
  for (auto &input_arg : input_args) {
    MS_EXCEPTION_IF_NULL(input_arg);
  }
  const int64_t kInputsNum = 2;
  auto rank = SizeToLong(input_args.size());
  MS_CHECK_VALUE(rank >= kInputsNum, CheckAndConvertUtils::FormatCheckIntegerMsg("Input numbers", rank, kGreaterEqual,
                                                                                 kInputsNum, primitive));
  auto x_shape_ptr = input_args[kInputIndex0]->GetShape();
  MS_EXCEPTION_IF_NULL(x_shape_ptr);
  auto x_shape = x_shape_ptr->GetShapeVector();
  auto y_shape_ptr = input_args[kInputIndex1]->GetShape();
  MS_EXCEPTION_IF_NULL(y_shape_ptr);
  auto y_shape = y_shape_ptr->GetShapeVector();
  auto output_shape = CalBroadCastShape(x_shape, y_shape, primitive->name());
  return std::make_shared<abstract::TensorShape>(output_shape);
}

TypePtr BinaryOpFuncImpl::InferType(const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  auto input_type = input_args[kInputIndex0]->GetType();
  MS_EXCEPTION_IF_NULL(input_type);
  return input_type->Clone();
}
}  // namespace ops
}  // namespace mindspore
