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
#include "ops/coalesce.h"

#include "abstract/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/dshape.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
TuplePtr CoalesceInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x_values", input_args[kInputIndex1]->BuildType(), valid_types,
                                                   prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x_indices", input_args[kInputIndex0]->BuildType(), {kInt64},
                                                   prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x_shape", input_args[kInputIndex2]->BuildType(), {kInt64},
                                                   prim->name());
  std::vector<TypePtr> types_list = {input_args[0]->BuildType(), input_args[1]->BuildType(),
                                     input_args[2]->BuildType()};
  return std::make_shared<Tuple>(types_list);
}

abstract::TupleShapePtr CoalesceInferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  constexpr int x_indices_shape_size = 2;
  auto x_indices_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto x_values_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto x_shape_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  if (x_indices_shape.size() != x_indices_shape_size || x_values_shape.size() != 1 || x_shape_shape.size() != 1) {
    MS_EXCEPTION(ValueError) << "For Coalesce, x_indices should be a 2-D tensor"
                             << ", x_values should be a 1-D tensor"
                             << ", x_shape should be a 1-D tensor"
                             << ", but got x_indices is a " << x_indices_shape.size() << "-D tensor"
                             << ", got x_values is a " << x_values_shape.size() << "-D tensor"
                             << ", got x_shape is a " << x_shape_shape.size() << "-D tensor";
  }
  if (x_indices_shape[0] != x_shape_shape[0]) {
    MS_EXCEPTION(ValueError) << "For " << prim_name
                             << ", sizes of dim0 of x_indices and dim0 of x_shape should be the same"
                             << ", but size of dim0 of got x_indices is " << x_indices_shape[0]
                             << ", size of dim0 of got x_shape is " << x_shape_shape[0];
  }
  if (x_indices_shape[1] != x_values_shape[0]) {
    MS_EXCEPTION(ValueError) << "For " << prim_name
                             << ", sizes of dim1 of x_indices and dim0 of x_values should be the same"
                             << ", but size of dim1 of got x_indices is " << x_indices_shape[1]
                             << ", size of dim0 of got x_values is " << x_values_shape[0];
  }
  ShapeVector y_indices_shape = {x_indices_shape[0], -1};
  ShapeVector y_indices_min_shape = {x_indices_shape[0], 1};
  ShapeVector y_indices_max_shape = {x_indices_shape[0], x_indices_shape[1]};
  ShapeVector y_values_shape = {-1};
  ShapeVector y_values_min_shape = {1};
  ShapeVector y_values_max_shape = {x_indices_shape[1]};
  auto y_shape = input_args[2]->BuildShape();
  MS_EXCEPTION_IF_NULL(y_shape);
  abstract::ShapePtr y_shape_shape_list = y_shape->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(y_shape_shape_list);
  abstract::ShapePtr y_indices_shape_list =
    std::make_shared<abstract::Shape>(y_indices_shape, y_indices_min_shape, y_indices_max_shape);
  abstract::ShapePtr y_values_shape_list =
    std::make_shared<abstract::Shape>(y_values_shape, y_values_min_shape, y_values_max_shape);
  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{y_indices_shape_list, y_values_shape_list, y_shape_shape_list});
}
}  // namespace

MIND_API_BASE_IMPL(Coalesce, PrimitiveC, BaseOperator);
AbstractBasePtr CoalesceInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args) {
  const int64_t input_num = 3;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = CoalesceInferType(primitive, input_args);
  auto infer_shape = CoalesceInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(Coalesce, prim::kPrimCoalesce, CoalesceInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
