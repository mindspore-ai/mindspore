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

#include "ops/matrix_diag_part.h"
#include <set>
#include "abstract/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "abstract/utils.h"

namespace mindspore {
namespace ops {
namespace {
const constexpr int64_t kShape2 = 2;
abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto input_shape = input_args[0]->BuildShape();
  auto shape_element = input_shape->cast<abstract::ShapePtr>();
  ShapeVector shape = shape_element->shape();
  ShapeVector min_shape = shape_element->shape();
  ShapeVector max_shape = shape_element->shape();
  max_shape[shape.size() - 1] = kShape2 * shape[shape.size() - 1] - 1;
  min_shape[shape.size() - 1] = 1;
  shape[shape.size() - 1] = abstract::Shape::SHP_ANY;
  return std::make_shared<abstract::Shape>(shape, min_shape, max_shape);
}

TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto infer_type = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(infer_type);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64, kInt8, kInt16, kInt32, kInt64};
  CheckAndConvertUtils::CheckTensorTypeValid("input", infer_type, valid_types, prim->name());
  return infer_type;
}
}  // namespace

AbstractBasePtr MatrixDiagPartInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) {
  return abstract::MakeAbstract(InferShape(primitive, input_args), InferType(primitive, input_args));
}
REGISTER_PRIMITIVE_EVAL_IMPL(MatrixDiagPartV3, prim::kPrimMatrixDiagPart, MatrixDiagPartInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
