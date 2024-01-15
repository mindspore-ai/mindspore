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

#include "ops/ops_func_impl/scalar_to_tensor.h"

#include <utility>
#include <memory>
#include <set>
#include "ops/ops_frontend_func_impl.h"
#include "ops/auto_generate/gen_ops_name.h"
#include "ops/op_utils.h"
#include "ops/op_name.h"
#include "ir/dtype.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr ScalarToTensorFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) const {
  ShapeVector out_shape;
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr ScalarToTensorFuncImpl::InferType(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  const int64_t input_len = 2;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual, input_len,
                                           op_name);
  const std::set<TypePtr> valid_types = {kBool,   kInt8,   kInt16,   kInt32,   kInt64,   kUInt8,     kUInt16,
                                         kUInt32, kUInt64, kFloat16, kFloat32, kFloat64, kComplex64, kComplex128};
  TypePtr dst_type{nullptr};
  if (input_args[kInputIndex1]->GetType()->isa<TypeNone>()) {
    auto attr = primitive->GetAttr("dtype");
    if (attr == nullptr) {
      attr = input_args[0]->GetType();
    }
    MS_EXCEPTION_IF_NULL(attr);
    if (!attr->isa<Type>()) {
      MS_EXCEPTION(TypeError) << "For '" << op_name << "the second input must be a `Type`, but got "
                              << attr->type_name();
    }
    dst_type = attr->isa<TensorType>() ? attr->cast_ptr<TensorType>()->element() : attr->cast<TypePtr>();
  } else {
    auto dtype_value = GetScalarValue<int64_t>(input_args[kInputIndex1]->GetValue());
    MS_CHECK_VALUE(dtype_value.has_value(),
                   CheckAndConvertUtils::FormatCommMsg("For primitive[", op_name,
                                                       "], the `dtype` should has valid value for static type."));
    dst_type = TypeIdToType(static_cast<TypeId>(dtype_value.value()));
  }
  (void)CheckAndConvertUtils::CheckSubClass("dtype", dst_type, valid_types, op_name);
  return std::make_shared<TensorType>(dst_type);
}

}  // namespace ops
}  // namespace mindspore
