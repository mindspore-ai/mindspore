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

#include "ops/matrix_inverse.h"
#include <set>

#include "ops/op_utils.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr MatrixInverseInferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto x_rank = SizeToLong(x_shape.size());
  const constexpr int64_t kNumber1 = 1;
  const constexpr int64_t kNumber2 = 2;
  CheckAndConvertUtils::CheckInteger("x rank", x_rank, kGreaterEqual, kNumber2, prim_name);
  CheckAndConvertUtils::Check("row size", x_shape[x_rank - kNumber1], kEqual, x_shape[x_rank - kNumber2], prim_name);
  CheckAndConvertUtils::CheckInteger("row size", x_shape[x_rank - kNumber1], kGreaterEqual, kNumber2, prim_name);
  CheckAndConvertUtils::CheckInteger("column size", x_shape[x_rank - kNumber2], kGreaterEqual, kNumber2, prim_name);
  return std::make_shared<abstract::Shape>(x_shape);
}

TypePtr MatrixInverseInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::set<TypePtr> valid_types = {kFloat32, kFloat64};
  auto infer_type = input_args[0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", infer_type, valid_types, prim->name());
  return infer_type;
}
}  // namespace

MIND_API_BASE_IMPL(MatrixInverse, PrimitiveC, BaseOperator);
AbstractBasePtr MatrixInverseInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infertype = MatrixInverseInferType(primitive, input_args);
  auto infershape = MatrixInverseInferShape(primitive, input_args);
  return abstract::MakeAbstract(infershape, infertype);
}
REGISTER_PRIMITIVE_EVAL_IMPL(MatrixInverse, prim::kPrimMatrixInverse, MatrixInverseInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
