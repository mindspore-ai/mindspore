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
#include "ops/grad/grid_sampler_3d_grad.h"

namespace mindspore {
namespace ops {
namespace {
const size_t kZero = 0;
const size_t kOne = 1;
const size_t kTwo = 2;
const size_t kThree = 3;
const size_t kFour = 4;
const size_t kFive = 5;

abstract::TupleShapePtr GridSampler3DGradInferShape(const PrimitivePtr &primitive,
                                                    const std::vector<AbstractBasePtr> &input_args) {
  auto grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kZero]->BuildShape())[kShape];
  auto input_x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kOne]->BuildShape())[kShape];
  auto grid_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kTwo]->BuildShape())[kShape];
  if (grad_shape.size() != kFive) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', Grad must be a 5-dimensional tensor, but got "
                             << std::to_string(grad_shape.size()) << "-dimensional tensor.";
  }
  if (input_x_shape.size() != kFive) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', Input_x must be a 5-dimensional tensor, but got "
                             << std::to_string(input_x_shape.size()) << "-dimensional tensor.";
  }
  if (grid_shape.size() != kFive) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', Grid must be a 5-dimensional tensor, but got "
                             << std::to_string(grid_shape.size()) << "-dimensional tensor.";
  }
  if (input_x_shape[kZero] != grid_shape[kZero]) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', The shape of grid is "
                             << input_args[kTwo]->BuildShape()->ToString() << " , but the shape of input_x is "
                             << input_args[kOne]->BuildShape()->ToString()
                             << " . The first dimension of grid and input_x must be equal.";
  }
  if (grid_shape[kFour] != kThree) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', The last dimension of grid must be 3, but got "
                             << std::to_string(grid_shape[kFour]);
  }
  std::vector<int64_t> out_shape = {input_x_shape[kZero], input_x_shape[kOne], grid_shape[kOne], grid_shape[kTwo],
                                    grid_shape[kThree]};
  bool shape_error = false;
  for (size_t i = kZero; i < kFive; i++) {
    if (out_shape[i] != grad_shape[i]) {
      shape_error = true;
      break;
    }
  }
  if (shape_error) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', The shape of grad, which is the same as that of output, is "
                             << input_args[kZero]->BuildShape()->ToString() << ", but the shape of output is ("
                             << std::to_string(out_shape[kZero]) << ", " << std::to_string(out_shape[kOne]) << ", "
                             << std::to_string(out_shape[kTwo]) << ", " << std::to_string(out_shape[kThree]) << ", "
                             << std::to_string(out_shape[kFour]) << ").";
  }
  abstract::ShapePtr dx_shape = std::make_shared<abstract::Shape>(input_x_shape);
  abstract::ShapePtr dgrid_shape = std::make_shared<abstract::Shape>(grid_shape);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{dx_shape, dgrid_shape});
}

TuplePtr GridSampler3DGradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  std::map<std::string, TypePtr> types;
  std::set<TypePtr> valid_types = {kFloat32, kFloat64};
  TypePtr grad_type = input_args[kZero]->BuildType();
  TypePtr input_x_type = input_args[kOne]->BuildType();
  TypePtr grid_type = input_args[kTwo]->BuildType();
  (void)types.emplace("grad", grad_type);
  (void)types.emplace("input_x", input_x_type);
  (void)types.emplace("grid", grid_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, primitive->name());
  return std::make_shared<Tuple>(std::vector<TypePtr>{input_x_type, grid_type});
}
}  // namespace

AbstractBasePtr GridSampler3DGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = kThree;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_types = GridSampler3DGradInferType(primitive, input_args);
  auto infer_shapes = GridSampler3DGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shapes, infer_types);
}

REGISTER_PRIMITIVE_EVAL_IMPL(GridSampler3DGrad, prim::kPrimGridSampler3DGrad, GridSampler3DGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
