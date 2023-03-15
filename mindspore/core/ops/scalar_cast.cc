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
#include <set>
#include <memory>

#include "ops/scalar_cast.h"
#include "ops/op_utils.h"
#include "abstract/ops/op_infer.h"
#include "utils/check_convert_utils.h"
#include "include/common/utils/utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
template <typename T>
T GetTensorValue(const std::string &op_name, const tensor::TensorPtr &elem) {
  MS_EXCEPTION_IF_NULL(elem);
  TypeId type_id = elem->data_type();
  auto elem_c = elem->data_c();
  switch (type_id) {
    case kNumberTypeBool:
      return static_cast<T>(reinterpret_cast<bool *>(elem_c)[0]);
    case kNumberTypeInt8:
      return static_cast<T>(reinterpret_cast<int8_t *>(elem_c)[0]);
    case kNumberTypeInt16:
      return static_cast<T>(reinterpret_cast<int16_t *>(elem_c)[0]);
    case kNumberTypeInt32:
      return static_cast<T>(reinterpret_cast<int32_t *>(elem_c)[0]);
    case kNumberTypeInt64:
      return static_cast<T>(reinterpret_cast<int64_t *>(elem_c)[0]);
    case kNumberTypeUInt8:
      return static_cast<T>(reinterpret_cast<uint8_t *>(elem_c)[0]);
    case kNumberTypeUInt16:
      return static_cast<T>(reinterpret_cast<uint16_t *>(elem_c)[0]);
    case kNumberTypeUInt32:
      return static_cast<T>(reinterpret_cast<uint32_t *>(elem_c)[0]);
    case kNumberTypeUInt64:
      return static_cast<T>(reinterpret_cast<uint64_t *>(elem_c)[0]);
    case kNumberTypeFloat32:
      return static_cast<T>(reinterpret_cast<float *>(elem_c)[0]);
    case kNumberTypeFloat64:
      return static_cast<T>(reinterpret_cast<double *>(elem_c)[0]);
    default:
      MS_EXCEPTION(TypeError) << "For op '" << op_name << "' input must be number, but got " << elem->ToString();
  }
}

void CheckInputValid(const AbstractBasePtr &elem_x, const std::string &op_name) {
  if (!elem_x->isa<abstract::AbstractScalar>() && !elem_x->isa<abstract::AbstractTensor>()) {
    MS_EXCEPTION(TypeError) << "For '" << op_name
                            << "', the input should be scalar or tensor but got x: " << elem_x->ToString();
  }

  if (elem_x->isa<abstract::AbstractTensor>()) {
    std::vector<AbstractBasePtr> input_args;
    input_args.push_back(elem_x);
    auto shape_ptr = CheckAndConvertUtils::GetTensorInputShape(op_name, input_args, 0);
    MS_EXCEPTION_IF_NULL(shape_ptr);
    auto x_shape = shape_ptr->shape();
    if (!x_shape.empty() && !IsDynamic(x_shape) && !(x_shape.size() == 1 && x_shape[0] == 1)) {
      MS_EXCEPTION(ValueError) << "For Primitive[" << op_name << "], the input shape must be empty or 1, but got "
                               << x_shape << ".";
    }
  }
}

template <typename T>
T GetRealValue(const ValuePtr &x_value, const std::string &op_name, const bool &is_tensor) {
  MS_EXCEPTION_IF_NULL(x_value);
  T res;
  if (is_tensor) {
    res = GetTensorValue<T>(op_name, x_value->cast<tensor::TensorPtr>());
  } else {
    res = GetScalarValue<T>(op_name, x_value);
  }
  return res;
}

class ScalarCastInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto op_name = primitive->name();
    auto elem_x = input_args[0];
    CheckInputValid(elem_x, op_name);
    return abstract::kNoShape;
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto op_name = primitive->name();
    constexpr size_t input_len = 1;
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual, input_len,
                                             op_name);
    auto elem_x = input_args[0];
    if (!elem_x->isa<abstract::AbstractScalar>() && !elem_x->isa<abstract::AbstractTensor>()) {
      MS_EXCEPTION(TypeError) << "For '" << op_name
                              << "', the input should be scalar or tensor but got x: " << elem_x->ToString();
    }
    auto attr = primitive->GetAttr("dtype");
    if (attr == nullptr) {
      auto type_abs = abstract::CheckArg<abstract::AbstractType>(op_name, input_args, 1);
      attr = type_abs->BuildValue();
      MS_EXCEPTION_IF_NULL(attr);
    }
    if (!attr->isa<Type>()) {
      MS_EXCEPTION(TypeError) << "For '" << op_name << "the second input must be a `Type`, but got "
                              << attr->type_name();
    }
    auto output_dtype = attr->cast<TypePtr>();

    const std::set<TypePtr> valid_types = {kBool,   kInt8,   kInt16,   kInt32,   kInt64,     kUInt8,     kUInt16,
                                           kUInt32, kUInt64, kFloat32, kFloat64, kComplex64, kComplex128};
    return CheckAndConvertUtils::CheckSubClass("dtype", output_dtype, valid_types, op_name);
  }

  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
    MS_EXCEPTION_IF_NULL(primitive);
    constexpr size_t input_num = 2;
    bool is_tensor = false;
    auto op_name = primitive->name();
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);
    for (const auto &item : input_args) {
      MS_EXCEPTION_IF_NULL(item);
    }
    constexpr size_t x_index = 0;
    auto elem_x = input_args[x_index];
    CheckInputValid(elem_x, op_name);
    if (elem_x->isa<abstract::AbstractTensor>()) {
      is_tensor = true;
    }

    auto x_value = elem_x->BuildValue();
    if (x_value == kValueAny) {
      return nullptr;
    }
    auto res_type = InferType(primitive, input_args);
    ValuePtr res;
    switch (res_type->type_id()) {
      case kNumberTypeInt8:
        return MakeValue(GetRealValue<int8_t>(x_value, op_name, is_tensor));
      case kNumberTypeInt16:
        return MakeValue(GetRealValue<int16_t>(x_value, op_name, is_tensor));
      case kNumberTypeInt32:
        return MakeValue(GetRealValue<int32_t>(x_value, op_name, is_tensor));
      case kNumberTypeInt64:
        return MakeValue(GetRealValue<int64_t>(x_value, op_name, is_tensor));
      case kNumberTypeUInt8:
        return MakeValue(GetRealValue<uint8_t>(x_value, op_name, is_tensor));
      case kNumberTypeUInt16:
        return MakeValue(GetRealValue<uint16_t>(x_value, op_name, is_tensor));
      case kNumberTypeUInt32:
        return MakeValue(GetRealValue<uint32_t>(x_value, op_name, is_tensor));
      case kNumberTypeUInt64:
        return MakeValue(GetRealValue<uint32_t>(x_value, op_name, is_tensor));
      case kNumberTypeFloat32:
        return MakeValue(GetRealValue<float>(x_value, op_name, is_tensor));
      case kNumberTypeFloat64:
        return MakeValue(GetRealValue<double>(x_value, op_name, is_tensor));
      case kNumberTypeBool:
        return MakeValue(GetRealValue<bool>(x_value, op_name, is_tensor));
      default: {
        MS_EXCEPTION(TypeError) << "For '" << op_name
                                << "', the supported type is in the list: [int32, int64, float, double, bool], but got "
                                << res_type->ToString() << ".";
      }
    }
  }
};
MIND_API_OPERATOR_IMPL(ScalarCast, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ScalarCast, prim::kPrimScalarCast, ScalarCastInfer, true);
}  // namespace ops
}  // namespace mindspore
