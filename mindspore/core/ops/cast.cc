/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "ops/cast.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace ops {
class CastInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto x = input_args[0];
    abstract::BaseShapePtr shape_ptr{nullptr};
    if (x->isa<abstract::AbstractTensor>()) {
      auto shape = x->BuildShape();
      MS_EXCEPTION_IF_NULL(shape);
      shape_ptr = shape->cast<abstract::ShapePtr>();
    } else if (x->isa<abstract::AbstractScalar>()) {
      shape_ptr = std::make_shared<abstract::Shape>(ShapeVector{});
    } else {
      MS_EXCEPTION(TypeError) << "For '" << primitive->name() << "', input should be a Tensor or a number.";
    }
    MS_EXCEPTION_IF_NULL(shape_ptr);
    return shape_ptr;
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    const auto prim_name = primitive->name();
    CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, 1, prim_name);
    MS_EXCEPTION_IF_NULL(input_args[0]);
    auto x_type = input_args[0]->BuildType();
    CheckAndConvertUtils::CheckTypeValid("x", x_type, common_valid_types_with_complex_and_bool, prim_name);

    ValuePtr dst_type = primitive->GetAttr(kDstType);
    if (dst_type == nullptr) {
      const int64_t kCastInputNumWithDtype = 2;
      CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kCastInputNumWithDtype, prim_name);
      dst_type = input_args[1]->BuildValue();
    }
    if ((dst_type == nullptr) || (!dst_type->isa<Type>())) {
      MS_EXCEPTION(TypeError) << "Invalid dtype";
    }
    return dst_type->cast<TypePtr>();
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    auto type = InferType(primitive, input_args);
    auto shape = InferShape(primitive, input_args);
    auto tensor_type = std::make_shared<TensorType>(type);
    return abstract::MakeAbstract(shape, tensor_type);
  }
};

MIND_API_OPERATOR_IMPL(Cast, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(Cast, prim::kPrimCast, CastInfer, false);
}  // namespace ops
}  // namespace mindspore
