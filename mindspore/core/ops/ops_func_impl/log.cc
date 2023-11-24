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
#include "ops/ops_frontend_func_impl.h"

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
}  // namespace

BaseShapePtr LogFuncImpl::InferShape(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kIndex0]->GetShape()->Clone();
}

TypePtr LogFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kIndex0]->GetType()->Clone();
}

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
    auto result_tensor = std::make_shared<tensor::Tensor>(dtype, shape);  // same shape and dtype
    auto x_datac = x_tensor->data_c();
    MS_EXCEPTION_IF_NULL(result_tensor);
    auto result_datac = result_tensor->data_c();
    switch (dtype) {
      case kNumberTypeFloat16: {
        ImpleLog<float16>(x_datac, result_datac, data_size);
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
          << "', the supported data types are ['float16', 'float32', 'float64', 'complex64', 'complex128'], but got "
          << x_tensor->ToString();
      }
    }
    return result_tensor;
  }
};

REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("Log", LogFrontendFuncImpl);
}  // namespace ops
}  // namespace mindspore
