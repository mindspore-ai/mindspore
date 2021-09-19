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

#include <set>
#include "ops/grid_sampler_3d.h"

namespace mindspore {
namespace ops {
namespace {
const size_t kZero = 0;
const size_t kOne = 1;
const size_t kTwo = 2;
const size_t kThree = 3;
const size_t kFour = 4;
const size_t kFive = 5;

abstract::ShapePtr GridSampler3DInferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  auto input_x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kZero]->BuildShape())[kShape];
  auto grid_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kOne]->BuildShape())[kShape];
  if (input_x_shape.size() != kFive) {
    MS_EXCEPTION(ValueError) << "Input_x must be a 5-dimensional tensor, but got "
                             << std::to_string(input_x_shape.size()) << "-dimensional tensor.";
  }
  if (grid_shape.size() != kFive) {
    MS_EXCEPTION(ValueError) << "Grid must be a 5-dimensional tensor, but got " << std::to_string(grid_shape.size())
                             << "-dimensional tensor.";
  }
  if (input_x_shape[kZero] != grid_shape[kZero]) {
    MS_EXCEPTION(ValueError) << "The shape of grid is " << input_args[kOne]->BuildShape()->ToString()
                             << " , but the shape of input_x is " << input_args[kZero]->BuildShape()->ToString()
                             << " . The first dimension of grid and input_x must be equal.";
  }
  if (grid_shape[kFour] != kThree) {
    MS_EXCEPTION(ValueError) << "The last dimension of grid must be 3, but got " << std::to_string(grid_shape[kFour]);
  }
  std::vector<int64_t> output_shape = {input_x_shape[kZero], input_x_shape[kOne], grid_shape[kOne], grid_shape[kTwo],
                                       grid_shape[kThree]};
  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr GridSampler3DInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  std::map<std::string, TypePtr> types;
  std::set<TypePtr> valid_types = {kFloat32, kFloat64};
  TypePtr input_x_type = input_args[kZero]->BuildType();
  TypePtr grid_type = input_args[kOne]->BuildType();
  (void)types.emplace("input_x", input_x_type);
  (void)types.emplace("grid", grid_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, primitive->name());
  return input_x_type;
}
}  // namespace

AbstractBasePtr GridSampler3DInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = kTwo;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = GridSampler3DInferType(primitive, input_args);
  auto infer_shape = GridSampler3DInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

REGISTER_PRIMITIVE_EVAL_IMPL(GridSampler3D, prim::kPrimGridSampler3D, GridSampler3DInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
