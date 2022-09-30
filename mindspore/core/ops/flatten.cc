/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include <set>
#include "ops/flatten.h"
#include "utils/check_convert_utils.h"
#include "ops/primitive_c.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void Flatten::Init(const int64_t axis) { this->set_axis(axis); }
void Flatten::set_axis(const int64_t axis) { (void)this->AddAttr(kAxis, api::MakeValue(axis)); }
int64_t Flatten::get_axis() const {
  auto value_ptr = this->GetAttr(kAxis);
  return GetValue<int64_t>(value_ptr);
}
namespace {
abstract::ShapePtr FlattenInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input args size", SizeToLong(input_args.size()), kGreaterEqual, 1,
                                           prim_name);
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto x_shape = shape_map[kShape];
  if (IsDynamicRank(x_shape)) {
    return std::make_shared<abstract::Shape>(
      std::vector<int64_t>{abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny});
  }
  int64_t prod = 1;
  size_t size = x_shape.size();
  for (size_t i = 1; i < size; i++) {
    if (x_shape[i] == -1) {
      prod = -1;
      break;
    }
    prod = prod * x_shape[i];
  }
  ShapeVector out_shape = {x_shape[0], prod};
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr FlattenInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input args size", SizeToLong(input_args.size()), kGreaterEqual, 1,
                                           prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto x_dtype = input_args[0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_dtype, common_valid_types_with_complex_and_bool, prim_name);
  return x_dtype;
}
}  // namespace

MIND_API_OPERATOR_IMPL(Flatten, BaseOperator);
AbstractBasePtr FlattenInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  auto infer_type = FlattenInferType(primitive, input_args);
  auto infer_shape = FlattenInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(Flatten, prim::kPrimFlatten, FlattenInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
