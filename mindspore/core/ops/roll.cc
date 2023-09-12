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

#include "ops/roll.h"

#include <memory>
#include <set>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/math_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
namespace {
inline std::vector<int64_t> GetRollAttr(const PrimitivePtr &primitive, const std::string &attr_name) {
  auto prim_name = primitive->name();
  auto value = primitive->GetAttr(attr_name);
  MS_EXCEPTION_IF_NULL(value);
  std::vector<int64_t> values{};
  if (value->isa<ValueSequence>()) {
    values = GetValue<std::vector<int64_t>>(value);
  } else if (value->isa<Int64Imm>()) {
    values.emplace_back(GetValue<int64_t>(value));
  } else {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', '" << attr_name
                             << "'should be an int64 number or an array of int64 numbers.";
  }
  return values;
}

abstract::ShapePtr RollInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto axis = GetRollAttr(primitive, kAxis);
  auto shift = GetRollAttr(primitive, kShift);
  if (axis.size() != shift.size() || shift.empty()) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', 'axis' and 'shift' must be not empty and have same size.";
  }
  const int64_t input_num = 1;
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, input_num,
                                           prim_name);
  auto x = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 0);
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x->BuildShape())[kShape];
  return std::make_shared<abstract::Shape>(x_shape);
}

TypePtr RollInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  MS_EXCEPTION_IF_CHECK_FAIL(!input_args.empty(), "input_args must not be empty.");
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const std::set<TypePtr> valid_types = {kFloat32, kFloat16, kInt16, kInt32,   kInt64,
                                         kUInt32,  kInt8,    kUInt8, kFloat64, kBool};
  auto infer_type = input_args[0]->BuildType();
  return CheckAndConvertUtils::CheckTensorTypeValid("x type", infer_type, valid_types, prim->name());
}
}  // namespace

std::vector<int64_t> Roll::get_axis() const {
  std::vector<int64_t> axis_me;
  axis_me = GetValue<std::vector<int64_t>>(GetAttr(kAxis));
  return axis_me;
}

std::vector<int64_t> Roll::get_shift() const {
  std::vector<int64_t> shift_me;
  shift_me = GetValue<std::vector<int64_t>>(GetAttr(kShift));
  return shift_me;
}

MIND_API_OPERATOR_IMPL(Roll, BaseOperator);
AbstractBasePtr RollInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  return abstract::MakeAbstract(RollInferShape(primitive, input_args), RollInferType(primitive, input_args));
}

// AG means auto generated
class MIND_API AGRollInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return RollInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return RollInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return RollInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Roll, prim::kPrimRoll, AGRollInfer, false);
}  // namespace ops
}  // namespace mindspore
