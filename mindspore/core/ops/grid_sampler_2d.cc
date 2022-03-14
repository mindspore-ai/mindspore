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

#include "ops/grid_sampler_2d.h"
#include <set>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr GridSampler2DInferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  auto input_x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto grid_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  const int64_t normal_shape_size = 4;
  const int64_t label3 = 3;
  const int64_t num2 = 2;
  if (input_x_shape.size() != normal_shape_size) {
    MS_EXCEPTION(ValueError) << "Input_x must be a 4-dimensional tensor, but got "
                             << std::to_string(input_x_shape.size()) << "-dimensional tensor.";
  }
  if (grid_shape.size() != normal_shape_size) {
    MS_EXCEPTION(ValueError) << "Grid must be a 4-dimensional tensor, but got " << std::to_string(grid_shape.size())
                             << "-dimensional tensor.";
  }
  if (input_x_shape[0] != grid_shape[0]) {
    MS_EXCEPTION(ValueError) << "The shape of grid is " << input_args[1]->BuildShape()->ToString()
                             << " , but the shape of input_x is " << input_args[0]->BuildShape()->ToString()
                             << " . The first dimension of grid and input_x must be equal.";
  }
  if (grid_shape[label3] != num2) {
    MS_EXCEPTION(ValueError) << "The forth dimension of grid must be 2, but got " << std::to_string(grid_shape[label3]);
  }
  std::vector<int64_t> output_shape = {input_x_shape[0], input_x_shape[1], grid_shape[1], grid_shape[2]};
  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr GridSampler2DInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  if (std::any_of(input_args.begin(), input_args.end(), [](AbstractBasePtr arg) { return arg == nullptr; })) {
    MS_LOG(EXCEPTION) << "nullptr";
  }
  std::map<std::string, TypePtr> types;
  std::set<TypePtr> valid_types = {kFloat32};
  TypePtr input_x_type = input_args[0]->BuildType();
  TypePtr grid_type = input_args[1]->BuildType();
  (void)types.emplace("input_x", input_x_type);
  (void)types.emplace("grid", grid_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, primitive->name());
  if (input_x_type->type_id() != grid_type->type_id()) {
    MS_EXCEPTION(TypeError) << "The type of input_x must be the same as that of grid.";
  }
  return input_x_type;
}
}  // namespace

MIND_API_BASE_IMPL(GridSampler2D, PrimitiveC, BaseOperator);
AbstractBasePtr GridSampler2DInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = kInputIndex2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = GridSampler2DInferType(primitive, input_args);
  auto infer_shape = GridSampler2DInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

REGISTER_PRIMITIVE_EVAL_IMPL(GridSampler2D, prim::kPrimGridSampler2D, GridSampler2DInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
