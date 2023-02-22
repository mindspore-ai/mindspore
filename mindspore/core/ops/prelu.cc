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

#include "ops/prelu.h"
#include <set>
#include <map>
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"
#include "utils/ms_context.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
bool IsAscend() {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  return context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice;
}

abstract::ShapePtr PReLUInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto x_shape_ptr = input_args[kInputIndex0]->BuildShape();
  auto weight_shape_ptr = input_args[kInputIndex1]->BuildShape();
  // Dynamic rank.
  if (x_shape_ptr->IsDynamic() || weight_shape_ptr->IsDynamic()) {
    return x_shape_ptr->cast<abstract::ShapePtr>();
  }

  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x_shape_ptr)[kShape];
  auto weight_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(weight_shape_ptr)[kShape];
  auto x_rank = x_shape.size();
  auto weight_rank = weight_shape.size();
  auto channel_num = x_rank <= 1 ? 1 : x_shape[1];
  if (IsAscend() && x_rank <= 1) {
    MS_EXCEPTION(ValueError)
      << "For '" << prim_name
      << "', the dimension of 'x' can not be 0-D or 1-D when the platform is \"Ascend\", but got dimension of 'x' is "
      << x_rank << ".";
  }
  (void)CheckAndConvertUtils::CheckInteger("dimension of 'weight'", SizeToLong(weight_rank), kEqual, 1, prim_name);
  if (weight_shape[0] != 1 && weight_shape[0] != channel_num) {
    MS_EXCEPTION(ValueError)
      << "For '" << prim_name
      << "', the first dimension of 'weight' must be (1,) or it must be equal to number of channels: " << channel_num
      << ", but got " << weight_shape << ".";
  }
  return x_shape_ptr->cast<abstract::ShapePtr>();
}

TypePtr PReLUInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto x_type = input_args[kInputIndex0]->BuildType();
  auto weight_type = input_args[kInputIndex1]->BuildType();
  auto valid_types = {kFloat16, kFloat32};

  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("weight", weight_type, valid_types, prim_name);

  return x_type;
}

MIND_API_OPERATOR_IMPL(PReLU, BaseOperator);
AbstractBasePtr PReLUInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  constexpr int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto type = PReLUInferType(primitive, input_args);
  auto shape = PReLUInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}

// AG means auto generated
class MIND_API AGPReLUInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return PReLUInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return PReLUInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return PReLUInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(PReLU, prim::kPrimPReLU, AGPReLUInfer, false);
}  // namespace ops
}  // namespace mindspore
