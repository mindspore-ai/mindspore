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
#include "ops/prelu.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto x = input_args[0]->BuildShape();
  auto w = input_args[1]->BuildShape();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x)[kShape];
  auto w_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(w)[kShape];
  const int64_t x_rank = 2;
  const int64_t w_rank = 1;
  (void)CheckAndConvertUtils::CheckInteger("x rank", SizeToLong(x_shape.size()), kGreaterEqual, x_rank, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("weight rank", SizeToLong(w_shape.size()), kEqual, w_rank, prim_name);
  if (w_shape[0] != x_shape[1] && w_shape[0] != 1) {
    MS_LOG(EXCEPTION) << "For " << prim_name << ", channel of input_x and weight must be matched, "
                      << "while channel of input_x is " << x_shape[1] << ", weight_shape[0] is " << w_shape[0];
  }
  MS_EXCEPTION_IF_NULL(x);
  auto shape_element = x->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_element);
  return shape_element;
}

TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  std::map<string, TypePtr> check_map = {{"input_x", input_args[0]->BuildType()},
                                         {"weight", input_args[1]->BuildType()}};
  return CheckAndConvertUtils::CheckTensorTypeSame(check_map, valid_types, prim->name());
}
}  // namespace
AbstractBasePtr PReLUInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  return std::make_shared<abstract::AbstractTensor>(InferType(primitive, input_args),
                                                    InferShape(primitive, input_args));
}
REGISTER_PRIMITIVE_C(kNamePReLU, PReLU);
}  // namespace ops
}  // namespace mindspore
