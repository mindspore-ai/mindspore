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

#include "ops/grid_sampler_2d.h"

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
#include "ir/dtype/number.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr GridSampler2DInferShape(const PrimitivePtr &, const std::vector<AbstractBasePtr> &input_args) {
  auto input_x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto grid_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  const int64_t normal_shape_size = 4;
  const int64_t label3 = 3;
  const int64_t num2 = 2;
  // dynamic rank
  if (IsDynamicRank(input_x_shape) || IsDynamicRank(grid_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
  }
  // dynamic shape
  if (IsDynamic(input_x_shape) || IsDynamic(grid_shape)) {
    ShapeVector out_shape_dyn;
    for (size_t i = 0; i < input_x_shape.size(); ++i) {
      out_shape_dyn.push_back(abstract::Shape::kShapeDimAny);
    }
    return std::make_shared<abstract::Shape>(out_shape_dyn);
  }
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
  std::map<std::string, TypePtr> types;
  std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
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

MIND_API_OPERATOR_IMPL(GridSampler2D, BaseOperator);
AbstractBasePtr GridSampler2DInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = kInputIndex2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = GridSampler2DInferType(primitive, input_args);
  auto infer_shape = GridSampler2DInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

std::string GridSampler2D::get_interpolation_mode() const {
  auto value_ptr = this->GetAttr("interpolation_mode");
  return GetValue<std::string>(value_ptr);
}

std::string GridSampler2D::get_padding_mode() const {
  auto value_ptr = this->GetAttr("padding_mode");
  return GetValue<std::string>(value_ptr);
}

bool GridSampler2D::get_align_corners() const {
  auto value_ptr = this->GetAttr("align_corners");
  return GetValue<bool>(value_ptr);
}

// AG means auto generated
class MIND_API AGGridSampler2DInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return GridSampler2DInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return GridSampler2DInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return GridSampler2DInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(GridSampler2D, prim::kPrimGridSampler2D, AGGridSampler2DInfer, false);
}  // namespace ops
}  // namespace mindspore
