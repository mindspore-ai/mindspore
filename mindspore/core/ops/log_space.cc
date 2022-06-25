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

#include "ops/log_space.h"
#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr LogSpaceInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto start_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto start_shape = start_shape_map[kShape];
  if (start_shape.size() != 0) {
    MS_EXCEPTION(ValueError) << "For LogSpace, the dim of input[start] must be 0.";
  }
  auto end_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape());
  auto end_shape = end_shape_map[kShape];
  if (end_shape.size() != 0) {
    MS_EXCEPTION(ValueError) << "For LogSpace, the dim of input[end] must be 0.";
  }
  int64_t shape_value = GetValue<int64_t>(primitive->GetAttr("steps"));
  std::vector<int64_t> state_shape = {shape_value};
  return std::make_shared<abstract::Shape>(state_shape);
}

TypePtr LogSpaceInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("start", input_args[0]->BuildType());
  (void)types.emplace("end", input_args[1]->BuildType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
  auto dtype_attr = prim->GetAttr("dtype");
  MS_EXCEPTION_IF_NULL(dtype_attr);
  auto infer_type = dtype_attr->cast<TypePtr>();
  MS_EXCEPTION_IF_NULL(infer_type);
  return infer_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(LogSpace, BaseOperator);
AbstractBasePtr LogSpaceInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string op_name = primitive->name();
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, input_num, op_name);
  auto infer_type = LogSpaceInferType(primitive, input_args);
  auto infer_shape = LogSpaceInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(LogSpace, prim::kPrimLogSpace, LogSpaceInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
