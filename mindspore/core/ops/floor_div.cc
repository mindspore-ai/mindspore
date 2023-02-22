/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "ops/floor_div.h"

#include <string>
#include <vector>
#include <memory>
#include <set>

#include "abstract/abstract_value.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/primitive_c.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr FloorDivInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const int64_t max_dim = 8;
  auto in_shape_x = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto in_shape_y = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("The dimension of FloorDiv input", SizeToLong(in_shape_x.size()), kLessThan,
                                           max_dim, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("The dimension of FloorDiv input", SizeToLong(in_shape_y.size()), kLessThan,
                                           max_dim, prim_name);
  return BroadCastInferShape(prim_name, input_args);
}

TypePtr FloorDivInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  MS_EXCEPTION_IF_NULL(input_args[0]);
  MS_EXCEPTION_IF_NULL(input_args[1]);
  auto input_type01 = input_args[0]->BuildType();
  auto input_type02 = input_args[1]->BuildType();
  MS_EXCEPTION_IF_NULL(input_type01);
  MS_EXCEPTION_IF_NULL(input_type02);
  if (!input_type01->isa<TensorType>() && !input_type02->isa<TensorType>()) {
    MS_EXCEPTION(TypeError) << "For " << prim_name << ","
                            << " one of the inputs must be tensor type but got " << input_type01->ToString() << " and "
                            << input_type02->ToString() << ".";
  }
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64, kInt8,   kInt16, kInt32,     kInt64,
                                         kUInt8,   kUInt16,  kUInt32,  kUInt64, kBool,  kComplex64, kComplex128};
  (void)CheckAndConvertUtils::CheckTypeValid("x", input_type01, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTypeValid("y", input_type02, valid_types, prim_name);
  return input_type01;
}
}  // namespace

MIND_API_OPERATOR_IMPL(FloorDiv, BaseOperator);
AbstractBasePtr FloorDivInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infer_type = FloorDivInferType(primitive, input_args);
  auto infer_shape = FloorDivInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGFloorDivInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return FloorDivInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return FloorDivInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return FloorDivInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(FloorDiv, prim::kPrimFloorDiv, AGFloorDivInfer, false);
}  // namespace ops
}  // namespace mindspore
