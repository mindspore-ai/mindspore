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

#include "ops/grad/grid_sampler_2d_grad.h"

#include <set>
#include <map>

#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
const size_t KZero = 0;
const size_t KOne = 1;
const size_t KTwo = 2;
const size_t KThree = 3;
const size_t KFour = 4;

abstract::TupleShapePtr GridSampler2DGradInferShape(const PrimitivePtr &,
                                                    const std::vector<AbstractBasePtr> &input_args) {
  auto grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[KZero]->BuildShape())[kShape];
  auto input_x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[KOne]->BuildShape())[kShape];
  auto grid_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[KTwo]->BuildShape())[kShape];
  // dynamic rank
  if (IsDynamicRank(input_x_shape) || IsDynamicRank(grid_shape) || IsDynamicRank(grad_shape)) {
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{
      std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny}),
      std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny})});
  }
  if (grad_shape.size() != KFour) {
    MS_EXCEPTION(ValueError) << "Grad must be a 4-dimensional tensor, but got " << std::to_string(grad_shape.size())
                             << "-dimensional tensor.";
  }
  if (input_x_shape.size() != KFour) {
    MS_EXCEPTION(ValueError) << "Input_x must be a 4-dimensional tensor, but got "
                             << std::to_string(input_x_shape.size()) << "-dimensional tensor.";
  }
  if (grid_shape.size() != KFour) {
    MS_EXCEPTION(ValueError) << "Grid must be a 4-dimensional tensor, but got " << std::to_string(grid_shape.size())
                             << "-dimensional tensor.";
  }
  // dynamic shape
  if (IsDynamic(input_x_shape) || IsDynamic(grid_shape) || IsDynamic(grad_shape)) {
    ShapeVector dx_dyn_shape;
    ShapeVector dgrid_dyn_shape;
    for (size_t i = 0; i < input_x_shape.size(); ++i) {
      dx_dyn_shape.push_back(abstract::Shape::kShapeDimAny);
      dgrid_dyn_shape.push_back(abstract::Shape::kShapeDimAny);
    }
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{
      std::make_shared<abstract::Shape>(dx_dyn_shape), std::make_shared<abstract::Shape>(dgrid_dyn_shape)});
  }
  if (input_x_shape[KZero] != grid_shape[KZero]) {
    MS_EXCEPTION(ValueError) << "The shape of grid is " << input_args[KTwo]->BuildShape()->ToString()
                             << " , but the shape of input_x is " << input_args[KOne]->BuildShape()->ToString()
                             << " . The first dimension of grid and input_x must be equal.";
  }
  if (grid_shape[KThree] != SizeToLong(KTwo)) {
    MS_EXCEPTION(ValueError) << "The last dimension of grid must be 2, but got " << std::to_string(grid_shape[KThree]);
  }
  std::vector<int64_t> out_shape = {input_x_shape[KZero], input_x_shape[KOne], grid_shape[KOne], grid_shape[KTwo]};
  bool shape_error = false;
  for (size_t i = KZero; i < KFour; i++) {
    if (out_shape[i] != grad_shape[i]) {
      shape_error = true;
      break;
    }
  }
  if (shape_error) {
    MS_EXCEPTION(ValueError) << "The shape of grad, which is the same as that of output, is "
                             << input_args[KZero]->BuildShape()->ToString() << ", but the shape of output is ("
                             << std::to_string(out_shape[KZero]) << ", " << std::to_string(out_shape[KOne]) << ", "
                             << std::to_string(out_shape[KTwo]) << ", " << std::to_string(out_shape[KThree]) << ").";
  }
  abstract::ShapePtr dx_shape = std::make_shared<abstract::Shape>(input_x_shape);
  abstract::ShapePtr dgrid_shape = std::make_shared<abstract::Shape>(grid_shape);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{dx_shape, dgrid_shape});
}

TuplePtr GridSampler2DGradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  std::map<std::string, TypePtr> types;
  std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  TypePtr grad_type = input_args[KZero]->BuildType();
  TypePtr input_x_type = input_args[KOne]->BuildType();
  TypePtr grid_type = input_args[KTwo]->BuildType();
  (void)types.emplace("grad", grad_type);
  (void)types.emplace("input_x", input_x_type);
  (void)types.emplace("grid", grid_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, primitive->name());
  return std::make_shared<Tuple>(std::vector<TypePtr>{input_x_type, grid_type});
}
}  // namespace

MIND_API_OPERATOR_IMPL(GridSampler2DGrad, BaseOperator);
AbstractBasePtr GridSampler2DGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_types = GridSampler2DGradInferType(primitive, input_args);
  auto infer_shapes = GridSampler2DGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shapes, infer_types);
}

std::string GridSampler2DGrad::get_interpolation_mode() const {
  auto value_ptr = this->GetAttr("interpolation_mode");
  return GetValue<std::string>(value_ptr);
}

std::string GridSampler2DGrad::get_padding_mode() const {
  auto value_ptr = this->GetAttr("padding_mode");
  return GetValue<std::string>(value_ptr);
}

bool GridSampler2DGrad::get_align_corners() const {
  auto value_ptr = this->GetAttr("align_corners");
  return GetValue<bool>(value_ptr);
}

// AG means auto generated
class MIND_API AGGridSampler2DGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return GridSampler2DGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return GridSampler2DGradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return GridSampler2DGradInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(GridSampler2DGrad, prim::kPrimGridSampler2DGrad, AGGridSampler2DGradInfer, false);
}  // namespace ops
}  // namespace mindspore
