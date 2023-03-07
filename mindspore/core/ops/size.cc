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

#include "ops/size.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t input_num = 1;
}  // namespace
class SizeInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
    return abstract::kNoShape;
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
    TypePtr res = kInt64;
    return res;
  }

  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
    auto input_type = input_args[0]->BuildType();
    MS_EXCEPTION_IF_NULL(input_type);
    if (!input_type->isa<TensorType>()) {
      MS_EXCEPTION(TypeError) << "For '" << prim_name
                              << "', input must be a Tensor, but got: " << input_type->ToString() << ".";
    }
    auto input_shape_ptr = input_args[0]->BuildShape();
    MS_EXCEPTION_IF_NULL(input_shape_ptr);
    auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_shape_ptr);
    auto input_shape = shape_map[kShape];
    if (IsDynamicRank(input_shape) || IsDynamicShape(input_shape)) {
      return kAnyValue;
    }
    size_t elements = 1;
    for (size_t i = 0; i < input_shape.size(); i++) {
      elements *= input_shape[i];
    }
    auto elements_value = SizeToLong(elements);
    ValuePtr res = MakeValue(elements_value);
    return res;
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    auto type = InferType(primitive, input_args);
    auto shape = InferShape(primitive, input_args);
    auto value = InferValue(primitive, input_args);
    auto res = MakeAbstract(shape, type);
    res->set_value(value);
    return res;
  }
};
MIND_API_OPERATOR_IMPL(Size, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(Size, prim::kPrimSize, SizeInfer, true);
}  // namespace ops
}  // namespace mindspore
