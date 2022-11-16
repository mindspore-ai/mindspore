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

#include "ops/tridiagonal_solve.h"
#include <set>
#include <map>
#include <string>
#include <utility>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr TridiagonalSolveInferShape(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto diagonals_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto rhs_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape());
  auto diagonals_shp = diagonals_shape_map[kShape];
  auto rhs_shp = rhs_shape_map[kShape];
  int numberforthelastsecend = 2;
  int numberofdiagonals = 3;
  if (static_cast<int>(diagonals_shp.size()) <= 1) {
    MS_EXCEPTION(ValueError)
      << "For TridiagonalSolve, the dimensions of the input diagonals should be more than 1, but got "
      << diagonals_shp.size() << ".";
  }

  if (diagonals_shp.size() != rhs_shp.size()) {
    MS_EXCEPTION(ValueError) << "For TridiagonalSolve, expected the rank of diagonals and rhs to be the same, but got "
                             << diagonals_shp.size() << " and " << rhs_shp.size() << ".";
  }

  if (diagonals_shp[diagonals_shp.size() - numberforthelastsecend] != numberofdiagonals) {
    MS_EXCEPTION(ValueError)
      << "For TridiagonalSolve, the last second dimension of the input diagonals should be 3, but got "
      << diagonals_shp[diagonals_shp.size() - numberforthelastsecend] << ".";
  }

  if (diagonals_shp[diagonals_shp.size() - 1] != rhs_shp[rhs_shp.size() - numberforthelastsecend]) {
    MS_EXCEPTION(ValueError)
      << "For TridiagonalSolve, the last dimension of the input diagonals and the last second dimension of the input "
      << diagonals_shp[diagonals_shp.size() - 1] << " and " << rhs_shp[rhs_shp.size() - numberforthelastsecend] << ".";
  }

  return std::make_shared<abstract::Shape>(rhs_shp);
}

TypePtr TridiagonalSolveInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  const std::set<TypePtr> valid_types = {kFloat32, kFloat64, kComplex64, kComplex128};
  std::map<std::string, TypePtr> types;
  types.insert({"diagonals", input_args[0]->BuildType()});
  types.insert({"rhs", input_args[1]->BuildType()});
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(types, valid_types, prim->name());

  return input_args[1]->BuildType();
}
}  // namespace

MIND_API_OPERATOR_IMPL(TridiagonalSolve, BaseOperator);
AbstractBasePtr TridiagonalSolveInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = TridiagonalSolveInferType(primitive, input_args);
  auto infer_shape = TridiagonalSolveInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

REGISTER_PRIMITIVE_EVAL_IMPL(TridiagonalSolve, prim::kPrimTridiagonalSolve, TridiagonalSolveInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
