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

#include "ops/ops_func_impl/cast.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr CastFuncImpl::InferShape(const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) const {
  //  MS_EXCEPTION_IF_NULL(primitive);
  //  auto x = input_args[0];
  //  abstract::BaseShapePtr shape_ptr{nullptr};
  //  if (CheckAndConvertUtils::IsTensor(input_args[kInputIndex0])) {
  //    auto shape = x->GetShape();
  //    MS_EXCEPTION_IF_NULL(shape);
  //    shape_ptr = shape->cast<abstract::ShapePtr>();
  //  } else if (CheckAndConvertUtils::IsScalar(input_args[kInputIndex0])) {
  //    shape_ptr = std::make_shared<abstract::Shape>(ShapeVector{});
  //  } else {
  //    MS_EXCEPTION(TypeError) << "For '" << primitive->name() << "', input should be a Tensor or a number.";
  //  }
  //  MS_EXCEPTION_IF_NULL(shape_ptr);
  //  return shape_ptr;
  return nullptr;
}

TypePtr CastFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  //  MS_EXCEPTION_IF_NULL(primitive);
  //  const auto prim_name = primitive->name();
  //  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, 1, prim_name);
  //  MS_EXCEPTION_IF_NULL(input_args[0]);
  //  auto x_type = input_args[0]->GetType();
  //  (void)CheckAndConvertUtils::CheckTypeValid("x", x_type, common_valid_types_with_complex_and_bool, prim_name);
  //
  //  constexpr int64_t kCastInputNumWithDtype = 2;
  //  ValuePtr dst_type;
  //  if (input_args.size() == kCastInputNumWithDtype) {
  //    dst_type = input_args[1]->GetValue();
  //  } else {
  //    dst_type = primitive->GetAttr(kDstType);
  //  }
  //
  //  if ((dst_type == nullptr) || (!dst_type->isa<Type>())) {
  //    MS_EXCEPTION(TypeError) << "Invalid dtype";
  //  }
  //
  //  if (dst_type->isa<TensorType>()) {
  //    (void)primitive->AddAttr("DstT", dst_type->cast_ptr<TensorType>()->element());
  //  } else {
  //    (void)primitive->AddAttr("DstT", dst_type);
  //  }
  //
  //  if (x_type->isa<TensorType>()) {
  //    (void)primitive->AddAttr("SrcT", x_type->cast_ptr<TensorType>()->element());
  //  } else {
  //    (void)primitive->AddAttr("SrcT", x_type);
  //  }
  //  return dst_type->cast<TypePtr>();
  return nullptr;
}
}  // namespace ops
}  // namespace mindspore
