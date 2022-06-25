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

#include <map>
#include <set>
#include <string>

#include "ops/argmax_with_value.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr ArgMaxWithValueInferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) {
  auto in_shape_ptr = input_args[0]->BuildShape();
  auto in_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(in_shape_ptr)[kShape];
  auto axis = GetValue<int64_t>(primitive->GetAttr("axis"));
  auto keep_dims = GetValue<bool>(primitive->GetAttr("keep_dims"));
  auto in_rank = static_cast<int64_t>(in_shape.size());
  if (in_rank == 0) {
    if (axis != -1 && axis != 0) {
      MS_EXCEPTION(ValueError) << "For ArgMaxWithValue with 0d input tensor, axis must be one of 0 or -1, but got"
                               << axis << ".";
    }
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{in_shape_ptr, in_shape_ptr});
  }
  if (axis < 0) {
    axis += in_rank;
  }
  if (axis < 0 || axis >= in_rank) {
    MS_EXCEPTION(ValueError) << "For ArgMaxWithValue, axis must be in range [-in_rank, in_rank), but got" << axis
                             << ".";
  }
  auto output_shape = in_shape;
  if (keep_dims) {
    output_shape[axis] = 1;
  } else {
    (void)output_shape.erase(output_shape.begin() + axis);
  }

  (void)primitive->AddAttr("dimension", MakeValue(axis));
  auto index_shape = std::make_shared<abstract::Shape>(output_shape);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{index_shape, index_shape});
}

TuplePtr ArgMaxWithValueInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  TypePtr input_x_type = input_args[0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input_x", input_x_type, valid_types, prim->name());
  auto index_type = std::make_shared<TensorType>(kInt32);
  return std::make_shared<Tuple>(std::vector<TypePtr>{index_type, input_x_type});
}
}  // namespace
AbstractBasePtr ArgMaxWithValueInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto shapes = ArgMaxWithValueInferShape(primitive, input_args);
  auto types = ArgMaxWithValueInferType(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}
MIND_API_OPERATOR_IMPL(ArgMaxWithValue, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(ArgMaxWithValue, prim::kPrimArgMaxWithValue, ArgMaxWithValueInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
