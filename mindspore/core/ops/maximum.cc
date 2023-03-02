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

#include <map>
#include <string>
#include "ops/maximum.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
class MaximumInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    const int64_t kInputNum = 2;
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual, kInputNum,
                                             prim_name);
    for (const auto &item : input_args) {
      MS_EXCEPTION_IF_NULL(item);
    }
    return BroadCastInferShape(prim_name, input_args);
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    for (const auto &item : input_args) {
      MS_EXCEPTION_IF_NULL(item);
    }
    auto op_name = prim->name();
    const int64_t kInputNum = 2;
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual, kInputNum,
                                             op_name);
    std::map<std::string, TypePtr> types;
    (void)types.emplace("x", input_args[0]->BuildType());
    (void)types.emplace("y", input_args[1]->BuildType());

    auto type_x = input_args[0]->BuildType();
    auto type_y = input_args[1]->BuildType();
    MS_EXCEPTION_IF_NULL(type_x);
    MS_EXCEPTION_IF_NULL(type_y);
    if (type_x->isa<Complex>() || type_y->isa<Complex>()) {
      if (type_x->type_id() == kNumberTypeComplex64 && type_y->type_id() == kNumberTypeComplex64) {
        return type_x;
      } else if (type_x->type_id() == kNumberTypeComplex64 && type_y->type_id() == kNumberTypeFloat32) {
        return type_x;
      } else if (type_x->type_id() == kNumberTypeComplex128 && type_y->type_id() == kNumberTypeComplex128) {
        return type_x;
      } else if (type_x->type_id() == kNumberTypeComplex128 && type_y->type_id() == kNumberTypeFloat64) {
        return type_x;
      } else if (type_x->type_id() == kNumberTypeFloat32 && type_y->type_id() == kNumberTypeComplex64) {
        return type_y;
      } else if (type_x->type_id() == kNumberTypeFloat64 && type_y->type_id() == kNumberTypeComplex128) {
        return type_y;
      } else {
        MS_EXCEPTION(TypeError)
          << "For '" << op_name
          << "', complex math binary op expecting Tensor [complex64, complex64],[complex64, float32], [float32, "
             "complex64], [complex128, complex128], [complex128, float64] or [float64, complex128], but got ["
          << type_x->ToString() << ", " << type_y->ToString() << "].";
      }
    }
    (void)CheckAndConvertUtils::CheckTensorTypeSame(types, common_valid_types_with_complex_and_bool, prim->name());
    return type_x;
  }
};

MIND_API_OPERATOR_IMPL(Maximum, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(Maximum, prim::kPrimMaximum, MaximumInfer, false);
}  // namespace ops
}  // namespace mindspore
