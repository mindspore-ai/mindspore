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

#include "ops/cummax.h"
#include <map>
#include <string>
#include <set>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr CummaxInferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  auto x_shape = input_args[0]->BuildShape();
  auto x_shape_value = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x_shape)[kShape];
  auto dim = GetValue<int64_t>(primitive->GetAttr("dim"));
  if (x_shape_value.size() <= 0) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', Inputs should not be a " << x_shape_value.size()
                             << " dimensional tensor.";
  }
  if (dim >= static_cast<int64_t>(x_shape_value.size()) || dim < -static_cast<int64_t>(x_shape_value.size())) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "',The value of `dim` should be in the range of ["
                             << -static_cast<int64_t>(x_shape_value.size()) << ","
                             << static_cast<int64_t>(x_shape_value.size()) << ")";
  }
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{x_shape, x_shape});
}

TuplePtr CummaxInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  const std::set<TypePtr> valid_types = {kInt8, kInt32, kInt64, kUInt8, kUInt32, kFloat16, kFloat32};
  auto y_type = CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[0]->BuildType(), valid_types, op_name);
  auto indices_type = kInt64;
  return std::make_shared<Tuple>(std::vector<TypePtr>{y_type, indices_type});
}

AbstractBasePtr CummaxInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto types = CummaxInferType(primitive, input_args);
  auto shapes = CummaxInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}

REGISTER_PRIMITIVE_EVAL_IMPL(Cummax, prim::kPrimCummax, CummaxInfer, nullptr, true);
}  // namespace
}  // namespace ops
}  // namespace mindspore
