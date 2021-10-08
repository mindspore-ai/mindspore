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

#include "ops/ones.h"
#include <memory>
#include <set>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
// ones
namespace {
abstract::ShapePtr OnesInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  // check
  auto shape_value = input_args[0]->BuildValue();
  std::vector<int64_t> out_shape = CheckAndConvertUtils::CheckAttrIntOrTupleInt("shape", shape_value, prim_name);
  (void)CheckAndConvertUtils::CheckPositiveVector("shape", out_shape, prim_name);
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr OnesInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  // check
  auto dtype_value = input_args[1]->BuildValue();
  if (!dtype_value->isa<Type>()) {
    MS_EXCEPTION(TypeError) << "The dtype of Ones is invalid!";
  }
  auto output_type = dtype_value->cast<TypePtr>();
  const std::set<TypePtr> valid_types = {kBool,   kInt8,   kInt16,  kInt32,   kInt64,   kUInt8,
                                         kUInt16, kUInt32, kUInt64, kFloat16, kFloat32, kFloat64};
  return CheckAndConvertUtils::CheckSubClass("dtype", output_type, valid_types, prim_name);
}
AbstractBasePtr OnesInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string op_name = primitive->name();
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, input_num, op_name);

  return abstract::MakeAbstract(OnesInferShape(primitive, input_args), OnesInferType(primitive, input_args));
}

ValuePtr OnesInferValue(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto abs = OnesInfer(nullptr, prim, input_args);
  // check
  auto out_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(abs->BuildShape())[kShape];
  auto out_type = abs->BuildType();
  MS_EXCEPTION_IF_NULL(out_type);
  return TensorConstructUtils::CreateOnesTensor(out_type, out_shape);
}
}  // namespace
REGISTER_PRIMITIVE_EVAL_IMPL(Ones, prim::kPrimOnes, OnesInfer, OnesInferValue, false);
}  // namespace ops
}  // namespace mindspore
