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

#include "ops/zeros.h"

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
// zeros
namespace {
abstract::ShapePtr ZerosInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  // check
  auto shape_value = input_args[0]->BuildValue();
  MS_EXCEPTION_IF_NULL(shape_value);
  if (shape_value->isa<ValueList>()) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name << "' input must be a Int or a tuple of Int elements, but got "
                            << shape_value->ToString() << ".";
  }
  std::vector<int64_t> out_shape = CheckAndConvertUtils::CheckIntOrTupleInt("input[shape]", shape_value, prim_name);
  (void)CheckAndConvertUtils::CheckPositiveVector("shape", out_shape, prim_name);
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr ZerosInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  // check
  auto dtype_value = input_args[1]->BuildValue();
  MS_EXCEPTION_IF_NULL(dtype_value);
  if (!dtype_value->isa<Type>()) {
    MS_EXCEPTION(TypeError)
      << "For '" << prim_name
      << "', the supported data type is ['bool', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16','uint32', "
         "'uint64','float16', 'float32', 'float64'], but got an invalid dtype!";
  }
  auto output_type = dtype_value->cast<TypePtr>();
  const std::set<TypePtr> valid_types = {kBool,   kInt8,   kInt16,  kInt32,   kInt64,   kUInt8,
                                         kUInt16, kUInt32, kUInt64, kFloat16, kFloat32, kFloat64};
  return CheckAndConvertUtils::CheckSubClass("dtype", output_type, valid_types, prim_name);
}
AbstractBasePtr ZerosInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  return abstract::MakeAbstract(ZerosInferShape(primitive, input_args), ZerosInferType(primitive, input_args));
}

ValuePtr ZerosInferValue(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, input_num, prim->name());
  auto abs = ZerosInfer(nullptr, prim, input_args);
  // check
  MS_EXCEPTION_IF_NULL(abs);
  auto out_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(abs->BuildShape())[kShape];
  auto out_type = abs->BuildType();
  MS_EXCEPTION_IF_NULL(out_type);
  return TensorConstructUtils::CreateZerosTensor(out_type, out_shape);
}
}  // namespace

MIND_API_OPERATOR_IMPL(Zeros, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(Zeros, prim::kPrimZeros, ZerosInfer, ZerosInferValue, false);
}  // namespace ops
}  // namespace mindspore
