/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "ops/polar.h"

#include <map>
#include <string>
#include <set>

#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "mindapi/base/type_id.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr PolarInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto abs_shape = shape_map[kShape];
  auto angle_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  if (IsDynamicRank(abs_shape) && IsDynamicRank(angle_shape)) {
    return std::make_shared<abstract::Shape>(abs_shape);
  }
  if (abs_shape != angle_shape && !IsDynamic(abs_shape) && !IsDynamic(angle_shape)) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << ", shape of inputs should be the same, but get shape of input[0] : " << abs_shape
                             << " , shape of input[1] : " << angle_shape << " .";
  }
  auto output_shape = abs_shape;
  for (size_t idx = 0; idx < output_shape.size(); ++idx) {
    output_shape[idx] = (output_shape[idx] == -1) ? angle_shape[idx] : output_shape[idx];
  }
  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr PolarInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  std::map<std::string, TypePtr> types;
  auto prim_name = primitive->name();
  auto abs_input_type = input_args[0]->BuildType();
  auto angle_input_type = input_args[1]->BuildType();
  (void)types.emplace("abs", abs_input_type);
  (void)types.emplace("angle", angle_input_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, std::set<TypePtr>{kFloat32, kFloat64}, prim_name);
  auto abs_input_tensor = abs_input_type->cast<TensorTypePtr>();
  TypeId abs_input_tensor_id = abs_input_tensor->element()->type_id();
  return abs_input_tensor_id == kNumberTypeFloat32 ? std::make_shared<TensorType>(kComplex64)
                                                   : std::make_shared<TensorType>(kComplex128);
}
}  // namespace

AbstractBasePtr PolarInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = PolarInferType(primitive, input_args);
  auto infer_shape = PolarInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(Polar, BaseOperator);

// AG means auto generated
class MIND_API AGPolarInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return PolarInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return PolarInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return PolarInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Polar, prim::kPrimPolar, AGPolarInfer, false);
}  // namespace ops
}  // namespace mindspore
