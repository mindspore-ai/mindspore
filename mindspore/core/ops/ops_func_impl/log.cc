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
#include "ops/ops_func_impl/log.h"
#include <complex>
#include <memory>
#include <set>
#include "ops/ops_frontend_func_impl.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
namespace {
template <typename T>
void ImpleLog(void *origin, void *target, size_t size) {
  MS_EXCEPTION_IF_NULL(origin);
  MS_EXCEPTION_IF_NULL(target);
  auto origin_data = reinterpret_cast<T *>(origin);
  auto target_data = reinterpret_cast<T *>(target);
  for (size_t i = 0; i < size; ++i) {
    target_data[i] = static_cast<T>(log(static_cast<double>(origin_data[i])));
  }
}

template <typename T>
void ImpleComplexLog(void *origin, void *target, size_t size) {
  MS_EXCEPTION_IF_NULL(origin);
  MS_EXCEPTION_IF_NULL(target);
  auto origin_data = reinterpret_cast<T *>(origin);
  auto target_data = reinterpret_cast<T *>(target);
  for (size_t i = 0; i < size; ++i) {
    target_data[i] = static_cast<T>(log(origin_data[i]));
  }
}

template <typename T>
void ImpleLogInteger(void *origin, void *target, size_t size) {
  MS_EXCEPTION_IF_NULL(origin);
  MS_EXCEPTION_IF_NULL(target);
  auto origin_data = reinterpret_cast<T *>(origin);
  auto target_data = reinterpret_cast<float *>(target);
  for (size_t i = 0; i < size; ++i) {
    target_data[i] = static_cast<float>(log(static_cast<double>(origin_data[i])));
  }
}

TypeId GetOutputTypeId(const TypeId &input_type_id) {
  static std::set<TypeId> intergral_set = {kNumberTypeBool,  kNumberTypeUInt8, kNumberTypeInt8,
                                           kNumberTypeInt16, kNumberTypeInt32, kNumberTypeInt64};
  if (intergral_set.find(input_type_id) != intergral_set.end()) {
    return kNumberTypeFloat32;
  }
  return input_type_id;
}
}  // namespace

BaseShapePtr LogFuncImpl::InferShape(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kIndex0]->GetShape()->Clone();
}

TypePtr LogFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  auto input_type = input_args[kIndex0]->GetType();
  auto input_tensor_type = input_type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(input_tensor_type);
  auto input_type_id = input_tensor_type->element()->type_id();
  return std::make_shared<TensorType>(TypeIdToType(GetOutputTypeId(input_type_id)));
}

ShapeArray LogFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  return {x_tensor->shape()};
}

TypePtrList LogFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  return {TypeIdToType(GetOutputTypeId(x_tensor->Dtype()->type_id()))};
}
REGISTER_SIMPLE_INFER(kNameLog, LogFuncImpl)

class LogFrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  // Do not override this interface if the op has no InferValue
  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    auto x_value = input_args[kIndex0]->GetValue();
    if (x_value->ContainsValueAny()) {
      return nullptr;
    }
    auto x_tensor = x_value->cast<tensor::TensorPtr>();
    if (x_tensor == nullptr) {
      return nullptr;
    }
    MS_EXCEPTION_IF_NULL(x_tensor);
    auto data_size = x_tensor->DataSize();
    auto dtype = x_tensor->data_type();
    auto shape = input_args[kIndex0]->GetShape()->GetShapeVector();
    auto result_tensor = std::make_shared<tensor::Tensor>(GetOutputTypeId(dtype), shape);  // same shape and dtype
    auto x_datac = x_tensor->data_c();
    MS_EXCEPTION_IF_NULL(result_tensor);
    auto result_datac = result_tensor->data_c();
    switch (dtype) {
      case kNumberTypeBool: {
        ImpleLogInteger<bool>(x_datac, result_datac, data_size);
        break;
      }
      case kNumberTypeUInt8: {
        ImpleLogInteger<uint8_t>(x_datac, result_datac, data_size);
        break;
      }
      case kNumberTypeInt8: {
        ImpleLogInteger<int8_t>(x_datac, result_datac, data_size);
        break;
      }
      case kNumberTypeInt16: {
        ImpleLogInteger<int16_t>(x_datac, result_datac, data_size);
        break;
      }
      case kNumberTypeInt32: {
        ImpleLogInteger<int32_t>(x_datac, result_datac, data_size);
        break;
      }
      case kNumberTypeInt64: {
        ImpleLogInteger<int64_t>(x_datac, result_datac, data_size);
        break;
      }
      case kNumberTypeFloat16: {
        ImpleLog<float16>(x_datac, result_datac, data_size);
        break;
      }
      case kNumberTypeBFloat16: {
        ImpleLog<bfloat16>(x_datac, result_datac, data_size);
        break;
      }
      case kNumberTypeFloat32: {
        ImpleLog<float>(x_datac, result_datac, data_size);
        break;
      }
      case kNumberTypeFloat64: {
        ImpleLog<double>(x_datac, result_datac, data_size);
        break;
      }
      case kNumberTypeComplex64: {
        ImpleComplexLog<std::complex<float>>(x_datac, result_datac, data_size);
        break;
      }
      case kNumberTypeComplex128: {
        ImpleComplexLog<std::complex<double>>(x_datac, result_datac, data_size);
        break;
      }
      default: {
        MS_EXCEPTION(TypeError)
          << "For '" << primitive->name()
          << "', the supported data types are ['bool', 'uint8', 'int8', 'int16', 'int32', 'int64', 'float16', "
             "'bfloat16', 'float32', 'float64', 'complex64', 'complex128'], but got "
          << x_tensor->ToString();
      }
    }
    return result_tensor;
  }
};
REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL(kNameLog, LogFrontendFuncImpl);
}  // namespace ops
}  // namespace mindspore
