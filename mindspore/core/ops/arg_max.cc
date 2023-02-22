/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "ops/arg_max.h"

#include "mindapi/ir/type.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(Argmax, BaseOperator);
class ArgMaxAbsInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    const int64_t kArgMaxInputNum = 1;
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, kArgMaxInputNum,
                                             prim_name);
    for (const auto &item : input_args) {
      MS_EXCEPTION_IF_NULL(item);
    }
    auto x_base_shape = input_args[kInputIndex0]->BuildShape();
    MS_EXCEPTION_IF_NULL(x_base_shape);
    auto x_shape = x_base_shape->cast<abstract::ShapePtr>();
    MS_EXCEPTION_IF_NULL(x_shape);
    auto shape_vector = x_shape->shape();
    if (IsDynamicRank(shape_vector)) {
      return x_shape;
    }
    // Get rank of shape
    auto x_rank = shape_vector.size();
    // Get and calculate the real positive axis.
    auto axis_value = primitive->GetAttr(kAxis);
    auto axis = GetValue<int64_t>(axis_value);
    CheckAndConvertUtils::CheckInRange<int64_t>("axis", axis, kIncludeLeft, {-x_rank, x_rank}, prim_name);
    axis = axis < 0 ? axis + SizeToLong(x_rank) : axis;

    auto out_shape_vector = shape_vector;
    (void)out_shape_vector.erase(out_shape_vector.cbegin() + axis);
    return std::make_shared<abstract::Shape>(out_shape_vector);
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(prim);
    auto prim_name = prim->name();
    const int64_t kArgMaxInputNum = 1;
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, kArgMaxInputNum,
                                             prim_name);
    MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
    auto x = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex0);
    auto x_type = x->BuildType();
    MS_EXCEPTION_IF_NULL(x_type);
    if (!x_type->isa<TensorType>()) {
      MS_EXCEPTION(TypeError) << "For '" << prim_name << "', input must be a Tensor, but got: " << x_type->ToString()
                              << ".";
    }
    auto out_type = prim->GetAttr(kOutputType);
    MS_EXCEPTION_IF_NULL(out_type);
    auto type_ptr = out_type->cast<TypePtr>();
    MS_EXCEPTION_IF_NULL(type_ptr);
    return std::make_shared<TensorType>(type_ptr);
  }
};

void Argmax::Init(const int64_t axis, const TypeId output_type) {
  set_axis(axis);
  set_output_type(output_type);
}

void Argmax::set_axis(const int64_t axis) { (void)this->AddAttr(kAxis, api::MakeValue(axis)); }
void Argmax::set_output_type(const TypeId output_type) {
  (void)this->AddAttr(kOutputType, api::Type::GetType(output_type));
}

int64_t Argmax::get_axis() const { return GetValue<int64_t>(GetAttr(kAxis)); }
TypeId Argmax::get_output_type() const {
  auto type_ptr = GetAttr(kOutputType)->cast<api::TensorTypePtr>()->element();
  return type_ptr->type_id();
}

REGISTER_PRIMITIVE_OP_INFER_IMPL(Argmax, prim::kPrimArgMax, ArgMaxAbsInfer, false);
}  // namespace ops
}  // namespace mindspore
