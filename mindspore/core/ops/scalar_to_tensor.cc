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

#include "ops/scalar_to_tensor.h"

#include <vector>
#include <set>
#include <string>
#include <memory>

#include "abstract/ops/op_infer.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/param_validator.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ir/scalar.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/type_id.h"
#include "ops/core_ops.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
tensor::TensorPtr ScalarToTensorByType(const ScalarPtr &scalar, const TypePtr &src_type, const TypePtr &data_type) {
  MS_EXCEPTION_IF_NULL(scalar);
  MS_EXCEPTION_IF_NULL(data_type);
  MS_EXCEPTION_IF_NULL(src_type);
  TypeId type_id = src_type->type_id();
  switch (type_id) {
    case kNumberTypeBool:
      return std::make_shared<tensor::Tensor>(GetValue<bool>(scalar), data_type);
    case kNumberTypeUInt8:
      return std::make_shared<tensor::Tensor>(static_cast<uint64_t>(GetValue<uint8_t>(scalar)), data_type);
    case kNumberTypeUInt16:
      return std::make_shared<tensor::Tensor>(static_cast<uint64_t>(GetValue<uint16_t>(scalar)), data_type);
    case kNumberTypeUInt32:
      return std::make_shared<tensor::Tensor>(static_cast<uint64_t>(GetValue<uint32_t>(scalar)), data_type);
    case kNumberTypeUInt64:
      return std::make_shared<tensor::Tensor>(GetValue<uint64_t>(scalar), data_type);
    case kNumberTypeInt8:
      return std::make_shared<tensor::Tensor>(static_cast<int64_t>(GetValue<int8_t>(scalar)), data_type);
    case kNumberTypeInt16:
      return std::make_shared<tensor::Tensor>(static_cast<int64_t>(GetValue<int16_t>(scalar)), data_type);
    case kNumberTypeInt32:
      return std::make_shared<tensor::Tensor>(static_cast<int64_t>(GetValue<int32_t>(scalar)), data_type);
    case kNumberTypeInt64:
      return std::make_shared<tensor::Tensor>(GetValue<int64_t>(scalar), data_type);
    case kNumberTypeFloat32:
      return std::make_shared<tensor::Tensor>(GetValue<float>(scalar), data_type);
    case kNumberTypeFloat64:
      return std::make_shared<tensor::Tensor>(GetValue<double>(scalar), data_type);
    default:
      MS_LOG(EXCEPTION) << "When convert scalar to tensor, the scalar type: " << data_type << " is invalid.";
  }
}

class ScalarToTensorInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    ShapeVector out_shape;
    return std::make_shared<abstract::Shape>(out_shape);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    constexpr size_t input_len = 1;
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual, input_len,
                                             prim_name);
    auto elem = input_args[0];
    if (!elem->isa<abstract::AbstractScalar>()) {
      MS_EXCEPTION(TypeError) << "For '" << prim_name << "', the input should be scalar but got: " << elem->ToString();
    }

    auto attr = primitive->GetAttr("dtype");
    if (attr == nullptr) {
      auto type_abs = abstract::CheckArg<abstract::AbstractType>(prim_name, input_args, 1);
      attr = type_abs->BuildValue();
      MS_EXCEPTION_IF_NULL(attr);
    }
    if (!attr->isa<Type>()) {
      MS_EXCEPTION(TypeError) << "For '" << prim_name << "the second input must be a `Type`, but got "
                              << attr->type_name();
    }
    auto output_dtype = attr->cast<TypePtr>();

    const std::set<TypePtr> valid_types = {kBool,   kInt8,   kInt16,   kInt32,   kInt64,   kUInt8,     kUInt16,
                                           kUInt32, kUInt64, kFloat16, kFloat32, kFloat64, kComplex64, kComplex128};
    return std::make_shared<TensorType>(
      CheckAndConvertUtils::CheckSubClass("dtype", output_dtype, valid_types, prim_name));
  }

  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto op_name = primitive->name();
    constexpr size_t input_len = 1;
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual, input_len,
                                             op_name);
    auto elem = input_args[0];
    if (!elem->isa<abstract::AbstractScalar>()) {
      MS_EXCEPTION(TypeError) << "For '" << op_name << "', the input should be scalar but got: " << elem->ToString();
    }
    auto elem_value = elem->BuildValue();
    if (elem_value == kValueAny) {
      return nullptr;
    }
    if (!elem_value->isa<Scalar>()) {
      MS_EXCEPTION(TypeError) << "For '" << op_name
                              << "', the input should be scalar but got: " << elem_value->ToString();
    }
    TypePtr res_dtype;
    if (input_args.size() == 1) {
      res_dtype = std::make_shared<TensorType>(kFloat32);
    } else {
      res_dtype = InferType(primitive, input_args);
    }
    return ScalarToTensorByType(elem_value->cast<ScalarPtr>(), elem->BuildType(),
                                res_dtype->cast<TensorTypePtr>()->element());
  }
};
MIND_API_OPERATOR_IMPL(ScalarToTensor, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ScalarToTensor, prim::kPrimScalarToTensor, ScalarToTensorInfer, true);
}  // namespace ops
}  // namespace mindspore
