/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "ops/iou.h"
#include <algorithm>
#include <set>

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kGreaterEqual, 2, prim_name);
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 0);
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 1);
  auto x_shape_ptr = input_args[0]->BuildShape();
  MS_EXCEPTION_IF_NULL(x_shape_ptr);
  auto y_shape_ptr = input_args[1]->BuildShape();
  MS_EXCEPTION_IF_NULL(y_shape_ptr);
  auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x_shape_ptr);
  auto y_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(y_shape_ptr);
  auto x_shp = x_shape_map[kShape];
  auto y_shp = y_shape_map[kShape];
  if (x_shp.size() != 2 || y_shp.size() != 2) {
    MS_EXCEPTION(ValueError) << "For BatchMatMul, input x, y should have the same dimension size and should be greater"
                             << "or equal to 3, while x size = " << x_shp.size() << ", y size = " << y_shp.size();
  }
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(x_shp[1]), kGreaterEqual, 4, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(y_shp[1]), kGreaterEqual, 4, prim_name);
  ShapeVector x_min_shape = x_shape_map[kMinShape];
  ShapeVector x_max_shape = x_shape_map[kMaxShape];
  ShapeVector y_min_shape = y_shape_map[kMinShape];
  ShapeVector y_max_shape = y_shape_map[kMaxShape];
  ShapeVector ret_shape;
  ShapeVector ret_min_shape;
  ShapeVector ret_max_shape;
  ret_shape.push_back(y_shp[0]);
  ret_shape.push_back(x_shp[0]);
  if (y_shape_ptr->IsDynamic()) {
    ret_min_shape.push_back(y_min_shape[0]);
    ret_max_shape.push_back(y_max_shape[0]);
  } else {
    ret_min_shape.push_back(y_shp[0]);
    ret_max_shape.push_back(y_shp[0]);
  }
  if (x_shape_ptr->IsDynamic()) {
    ret_min_shape.push_back(x_min_shape[0]);
    ret_max_shape.push_back(x_max_shape[0]);
  } else {
    ret_min_shape.push_back(x_shp[0]);
    ret_max_shape.push_back(x_shp[0]);
  }
  return std::make_shared<abstract::Shape>(ret_shape, ret_min_shape, ret_max_shape);
}

TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[0]->BuildType());
  (void)types.emplace("y", input_args[1]->BuildType());
  return CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
}
}  // namespace
AbstractBasePtr IOUInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                         const std::vector<AbstractBasePtr> &input_args) {
  auto type = InferType(primitive, input_args);
  auto shape = InferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(IOU, prim::kPrimIOU, IOUInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
