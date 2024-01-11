/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#include <memory>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/masked_fill.h"

namespace mindspore::ops {
BaseShapePtr MaskedFillFuncImpl::InferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) const {
  auto op_name = primitive->name();
  auto input_shape = input_args[kIndex0]->GetShape()->GetShapeVector();
  auto mask_shape = input_args[kIndex1]->GetShape()->GetShapeVector();
  auto value_shape = input_args[kIndex2]->GetShape()->GetShapeVector();
  auto broadcast_shape = CalBroadCastShape(input_shape, mask_shape, op_name, "input", "mask");
  int64_t batch_rank = 0;

  if (primitive->HasAttr(kBatchRank)) {
    auto value_ptr = primitive->GetAttr(kBatchRank);
    batch_rank = GetValue<int64_t>(value_ptr);
  }
  if (batch_rank == 0 && value_shape.size() != 0) {
    MS_EXCEPTION(ValueError)
      << "For '" << op_name
      << "', 'value' only supports a 0-dimensional value tensor or a float number, but got tensor with "
      << value_shape.size() << " dimension(s).";
  } else if (value_shape.size() != 0) {
    (void)CheckAndConvertUtils::CheckInteger("value shape size", SizeToLong(value_shape.size()), kEqual, batch_rank,
                                             op_name);
    (void)CheckAndConvertUtils::CheckInteger("value shape size", SizeToLong(value_shape.size()), kLessEqual,
                                             SizeToLong(broadcast_shape.size()), op_name);
    for (size_t i = 0; i < LongToSize(batch_rank); i++) {
      if (value_shape[i] != broadcast_shape[i]) {
        MS_EXCEPTION(ValueError) << "For '" << op_name << "', the " << i
                                 << "th index of value shape should be equal to " << broadcast_shape[i] << ", but got "
                                 << value_shape[i];
      }
    }
  }

  return std::make_shared<abstract::Shape>(broadcast_shape);
}

TypePtr MaskedFillFuncImpl::InferType(const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("mask", input_args[kIndex1]->GetType(), {kBool}, prim_name);

  auto input_tensor_type = input_args[kIndex0]->GetType()->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(input_tensor_type);
  auto input_type_ptr = input_tensor_type->element();
  MS_EXCEPTION_IF_NULL(input_type_ptr);
  auto input_type = input_type_ptr->type_id();

  auto value_tensor_type = input_args[kIndex2]->GetType()->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(value_tensor_type);
  auto value_type_ptr = value_tensor_type->element();
  MS_EXCEPTION_IF_NULL(value_type_ptr);
  auto value_type = value_type_ptr->type_id();

  if (input_type != value_type) {
    MS_EXCEPTION(TypeError) << "For MaskedFill, "
                            << "the dtype of input and value should be same, but got: input's type "
                            << TypeIdToString(input_type) << ", value's type " << TypeIdToString(value_type) << ".";
  }
  return input_args[kIndex0]->GetType()->Clone();
}
}  // namespace mindspore::ops
