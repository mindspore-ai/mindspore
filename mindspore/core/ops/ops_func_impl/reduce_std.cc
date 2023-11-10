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

#include "ops/ops_func_impl/reduce_std.h"
#include "ops/ops_func_impl/reduce_arithmetic.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr ReduceStdFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  // For ReduceStd, the inputs order: x, axis, unbiased, keep_dims
  // For Reduce base infer function, the inputs order: x, axis, keep_dims
  MS_CHECK_VALUE(input_args.size() == 4, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                           "input_args number", SizeToLong(input_args.size()), kEqual, 4, primitive));
  auto tmp_input_args = std::vector<AbstractBasePtr>{input_args[0], input_args[1], input_args[3]};
  auto shape = ReduceInferShape(primitive, tmp_input_args);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>(2, shape));
}

TypePtr ReduceStdFuncImpl::InferType(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[0]);
  MS_EXCEPTION_IF_NULL(input_args[0]->GetType());
  return std::make_shared<Tuple>(std::vector<TypePtr>(2, input_args[0]->GetType()->Clone()));
}
}  // namespace ops
}  // namespace mindspore
