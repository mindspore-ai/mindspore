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

#include "ops/ops_func_impl/scalar_to_tensor.h"

#include <utility>
#include <memory>
#include "ops/ops_frontend_func_impl.h"
#include "ops/auto_generate/gen_ops_name.h"
#include "ops/op_utils.h"
#include "ops/op_name.h"
#include "ir/dtype.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
constexpr auto kScalarToTensor = "ScalarToTensor";
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
}  // namespace
class ScalarToTensorFrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto op_name = primitive->name();
    const int64_t input_len = 2;
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual, input_len,
                                             op_name);
    auto elem = input_args[0];
    if (!elem->isa<abstract::AbstractScalar>()) {
      MS_EXCEPTION(TypeError) << "For '" << op_name << "', the input should be scalar but got: " << elem->ToString();
    }
    auto elem_value = elem->GetValue();
    if (elem_value->ContainsValueAny()) {
      return nullptr;
    }
    if (!elem_value->isa<Scalar>()) {
      MS_EXCEPTION(TypeError) << "For '" << op_name
                              << "', the input should be scalar but got: " << elem_value->ToString();
    }
    TypePtr dst_type{nullptr};
    if (input_args[kInputIndex1]->GetType()->isa<TypeNone>()) {
      dst_type = std::make_shared<TensorType>(kFloat32);
    } else {
      ScalarToTensorFuncImpl funcImpl;
      dst_type = funcImpl.InferType(primitive, input_args);
    }
    return ScalarToTensorByType(elem_value->cast<ScalarPtr>(), elem->GetType(),
                                dst_type->cast<TensorTypePtr>()->element());
  }
};
REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL(kScalarToTensor, ScalarToTensorFrontendFuncImpl);
}  // namespace ops
}  // namespace mindspore
