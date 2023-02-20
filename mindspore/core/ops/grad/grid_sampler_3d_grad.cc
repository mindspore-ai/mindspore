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
#include <map>

#include "ops/grad/grid_sampler_3d_grad.h"
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
  // dynamic rank
  if (IsDynamicRank(input_x_shape) || IsDynamicRank(grid_shape) || IsDynamicRank(grad_shape)) {
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{
      std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny}),
      std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny})});
  }
  if (grad_shape.size() != kFive) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', 'grad' must be a 5-D tensor, but got a "
                             << std::to_string(grad_shape.size()) << "-D tensor.";
  }
  if (input_x_shape.size() != kFive) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', 'input_x' must be a 5-D tensor, but got a "
                             << std::to_string(input_x_shape.size()) << "-D tensor.";
  }
  if (grid_shape.size() != kFive) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', 'grid' must be a 5-D tensor, but got a "
                             << std::to_string(grid_shape.size()) << "-D tensor.";
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
  if (input_x_shape[kZero] != grid_shape[kZero]) {
    MS_EXCEPTION(ValueError)
      << "For '" << primitive->name()
      << "', the first dimension of 'grid' and 'input_x' must be equal. But got the shape of 'grid': "
      << input_args[kTwo]->BuildShape()->ToString()
      << ", the shape of 'input_x': " << input_args[kOne]->BuildShape()->ToString() << ".";
  }
  if (grid_shape[kFour] != SizeToLong(kThree)) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the last dimension of 'grid' must be 3, but got "
                             << std::to_string(grid_shape[kFour]) << ".";
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
                             << "', the shape of 'grad' must be the same as that of output, but got 'grad' shape: "
                             << input_args[kZero]->BuildShape()->ToString() << ", output shape: ("
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
  std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
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

MIND_API_OPERATOR_IMPL(GridSampler3DGrad, BaseOperator);
AbstractBasePtr GridSampler3DGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_types = GridSampler3DGradInferType(primitive, input_args);
  auto infer_shapes = GridSampler3DGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shapes, infer_types);
}

std::string GridSampler3DGrad::get_interpolation_mode() const {
  auto value_ptr = this->GetAttr("interpolation_mode");
  return GetValue<std::string>(value_ptr);
}

std::string GridSampler3DGrad::get_padding_mode() const {
  auto value_ptr = this->GetAttr("padding_mode");
  return GetValue<std::string>(value_ptr);
}

bool GridSampler3DGrad::get_align_corners() const {
  auto value_ptr = this->GetAttr("align_corners");
  return GetValue<bool>(value_ptr);
}

// AG means auto generated
class MIND_API AGGridSampler3DGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return GridSampler3DGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return GridSampler3DGradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return GridSampler3DGradInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(GridSampler3DGrad, prim::kPrimGridSampler3DGrad, AGGridSampler3DGradInfer, false);
}  // namespace ops
}  // namespace mindspore
