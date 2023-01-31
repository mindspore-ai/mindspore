/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include <vector>
#include <string>
#include <memory>
#include <set>
#include "ops/scalar_bool.h"

#include "ops/op_utils.h"
#include "abstract/ops/op_infer.h"
#include "utils/check_convert_utils.h"
#include "include/common/utils/utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
class ScalarBoolInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto op_name = primitive->name();
    constexpr size_t input_len = 1;
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_len, op_name);
    auto elem = input_args[0];
    if (!elem->isa<abstract::AbstractScalar>()) {
      MS_EXCEPTION(TypeError) << "For '" << op_name << "', the input should be scalar but got : " << elem->ToString();
    }
    return abstract::kNoShape;
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    auto x_type = input_args[0]->BuildType();
    std::set<TypePtr> check_types = {kInt32, kInt64, kFloat32, kFloat64, kBool};
    (void)CheckAndConvertUtils::CheckSubClass("x_dtype", x_type, check_types, prim_name);
    return kBool;
  }

  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
    MS_EXCEPTION_IF_NULL(primitive);
    constexpr size_t input_num = 1;
    auto op_name = primitive->name();
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);
    MS_EXCEPTION_IF_NULL(input_args[0]);
    auto elem = input_args[0];
    if (!elem->isa<abstract::AbstractScalar>()) {
      MS_EXCEPTION(TypeError) << "For '" << op_name << "', the input should be scalar but got : " << elem->ToString();
    }

    auto x_valueptr = elem->BuildValue();
    if (x_valueptr == kAnyValue) {
      return nullptr;
    }
    auto x_type = input_args[0]->BuildType();
    bool res;
    switch (x_type->type_id()) {
      case kNumberTypeInt32: {
        auto elem_value = GetValue<int32_t>(x_valueptr);
        res = static_cast<bool>(elem_value);
        break;
      }
      case kNumberTypeInt64: {
        auto elem_value = GetValue<int64_t>(x_valueptr);
        res = static_cast<bool>(elem_value);
        break;
      }
      case kNumberTypeFloat32: {
        auto elem_value = GetValue<float>(x_valueptr);
        res = static_cast<bool>(elem_value);
        break;
      }
      case kNumberTypeFloat64: {
        auto elem_value = GetValue<double>(x_valueptr);
        res = static_cast<bool>(elem_value);
        break;
      }
      case kNumberTypeBool: {
        auto elem_value = GetValue<bool>(x_valueptr);
        res = static_cast<bool>(elem_value);
      }
      default: {
        MS_EXCEPTION(TypeError)
          << "For '" << op_name
          << "', the supported type is in the list: [int32, int64, float32, float64, bool], but got "
          << x_type->ToString() << ".";
      }
    }
    return MakeValue(res);
  }
};
MIND_API_OPERATOR_IMPL(ScalarBool, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ScalarBool, prim::kPrimScalarBool, ScalarBoolInfer, true);
}  // namespace ops
}  // namespace mindspore
