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

#include "ops/linear_sum_assignment.h"

#include <map>
#include <set>
#include <algorithm>
#include <memory>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr LinearSumAssignmentInferShape(const PrimitivePtr &primitive,
                                                      const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  constexpr int64_t kNumber2 = 2;
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape());
  auto matrix_shape = shape_map[kShape];
  auto matrix_rank = SizeToLong(matrix_shape.size());
  (void)CheckAndConvertUtils::CheckInteger("cost_matrix rank", matrix_rank, kEqual, kNumber2, op_name);

  int64_t out_dim = std::min(matrix_shape[0], matrix_shape[1]);  // -1 or actual value
  ShapeVector row_ind_shape{1, out_dim};
  ShapeVector col_ind_shape{1, out_dim};
  std::vector<abstract::BaseShapePtr> shapes{std::make_shared<abstract::Shape>(row_ind_shape),
                                             std::make_shared<abstract::Shape>(col_ind_shape)};
  return std::make_shared<abstract::TupleShape>(shapes);
}

TuplePtr LinearSumAssignmentInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  const std::set<TypePtr> valid_cost_matrix_types = {kFloat32, kFloat64};
  const std::set<TypePtr> valid_dimention_limit_types = {kInt64};
  const std::set<TypePtr> valid_maximize_types = {kBool};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("cost_matrix", input_args[kInputIndex0]->BuildType(),
                                                   valid_cost_matrix_types, op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("dimension_limit", input_args[kInputIndex1]->BuildType(),
                                                   valid_dimention_limit_types, op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("maximize", input_args[kInputIndex2]->BuildType(),
                                                   valid_maximize_types, op_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{kInt64, kInt64});
}
}  // namespace

MIND_API_OPERATOR_IMPL(LinearSumAssignment, BaseOperator);

AbstractBasePtr LinearSumAssignmentInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  MS_EXCEPTION_IF_NULL(primitive);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  constexpr int64_t input_num = 3;
  (void)CheckAndConvertUtils::CheckInteger("Input numbers", SizeToLong(input_args.size()), kEqual, input_num, op_name);
  auto shape = LinearSumAssignmentInferShape(primitive, input_args);
  auto type = LinearSumAssignmentInferType(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}

// AG means auto generated
class MIND_API AGLinearSumAssignmentInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return LinearSumAssignmentInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return LinearSumAssignmentInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return LinearSumAssignmentInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(LinearSumAssignment, prim::kPrimLinearSumAssignment, AGLinearSumAssignmentInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
