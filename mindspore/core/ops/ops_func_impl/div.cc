/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include "ops/ops_func_impl/div.h"
#include <map>
#include <limits>
#include <string>
#include "utils/check_convert_utils.h"
#include "ops/ops_frontend_func_impl.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr DivFuncImpl::InferShape(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  return BroadCastInferShape(primitive->name(), input_args);
}

TypePtr DivFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  auto x_dtype = input_args[kIndex0]->GetType();
  auto y_dtype = input_args[kIndex1]->GetType();
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", x_dtype);
  (void)types.emplace("y", y_dtype);
  return CheckAndConvertUtils::CheckMathBinaryOpTensorType(types, common_valid_types_with_complex, prim_name);
}

template <typename T>
void DivImpl(void *x, void *y, void *result, size_t size) {
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(y);
  MS_EXCEPTION_IF_NULL(result);
  T *x_data = static_cast<T *>(x);
  T *y_data = static_cast<T *>(y);
  auto result_data = static_cast<T *>(result);
  MS_EXCEPTION_IF_NULL(x_data);
  MS_EXCEPTION_IF_NULL(y_data);
  MS_EXCEPTION_IF_NULL(result_data);
  auto zero = static_cast<T>(0);
  for (size_t i = 0; i < size; ++i) {
    if (y_data[i] == zero) {
      if (x_data[i] == zero) {
        result_data[i] = std::numeric_limits<T>::quiet_NaN();
        continue;
      }
      if (std::numeric_limits<T>::has_infinity) {
        result_data[i] = x_data[i] > zero ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
      } else {
        result_data[i] = x_data[i] > zero ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
      }
      continue;
    } else {
      result_data[i] = x_data[i] / y_data[i];
    }
  }
}

class DivFrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    DivFuncImpl div_infer_obj;
    auto result_type = div_infer_obj.InferType(primitive, input_args);
    auto result_shape = div_infer_obj.InferShape(primitive, input_args)->cast<abstract::ShapePtr>();
    auto x = input_args[kIndex0]->GetValue();
    auto y = input_args[kIndex1]->GetValue();
    if (x == nullptr || y == nullptr) {
      return nullptr;
    }
    if (x->ContainsValueAny() || y->ContainsValueAny()) {
      return nullptr;
    }
    std::cout << "x========" << x->ToString() << std::endl;
    std::cout << "y========" << y->ToString() << std::endl;
    auto x_tensor = x->cast<tensor::TensorPtr>();
    auto y_tensor = y->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(x_tensor);
    MS_EXCEPTION_IF_NULL(y_tensor);
    // auto type_id = x_tensor->data_type();
    // auto data_size = x_tensor->DataSize();
    auto type_id = result_type->type_id();
    auto result_tensor = std::make_shared<tensor::Tensor>(type_id, result_shape->shape());
    auto data_size = result_tensor->DataSize();
    switch (type_id) {
      case kNumberTypeBool: {
        DivImpl<bool>(x_tensor->data_c(), y_tensor->data_c(), result_tensor->data_c(), data_size);
        break;
      }
      case kNumberTypeInt: {
        DivImpl<int>(x_tensor->data_c(), y_tensor->data_c(), result_tensor->data_c(), data_size);
        break;
      }
      case kNumberTypeInt8: {
        DivImpl<int8_t>(x_tensor->data_c(), y_tensor->data_c(), result_tensor->data_c(), data_size);
        break;
      }
      case kNumberTypeInt16: {
        DivImpl<int16_t>(x_tensor->data_c(), y_tensor->data_c(), result_tensor->data_c(), data_size);
        break;
      }
      case kNumberTypeInt32: {
        DivImpl<int32_t>(x_tensor->data_c(), y_tensor->data_c(), result_tensor->data_c(), data_size);
        break;
      }
      case kNumberTypeInt64: {
        DivImpl<int64_t>(x_tensor->data_c(), y_tensor->data_c(), result_tensor->data_c(), data_size);
        break;
      }
      case kNumberTypeUInt: {
        DivImpl<uint32_t>(x_tensor->data_c(), y_tensor->data_c(), result_tensor->data_c(), data_size);
        break;
      }
      case kNumberTypeUInt8: {
        DivImpl<uint8_t>(x_tensor->data_c(), y_tensor->data_c(), result_tensor->data_c(), data_size);
        break;
      }
      case kNumberTypeUInt16: {
        DivImpl<uint16_t>(x_tensor->data_c(), y_tensor->data_c(), result_tensor->data_c(), data_size);
        break;
      }
      case kNumberTypeUInt32: {
        DivImpl<uint32_t>(x_tensor->data_c(), y_tensor->data_c(), result_tensor->data_c(), data_size);
        break;
      }
      case kNumberTypeUInt64: {
        DivImpl<uint64_t>(x_tensor->data_c(), y_tensor->data_c(), result_tensor->data_c(), data_size);
        break;
      }
      case kNumberTypeFloat: {
        DivImpl<float>(x_tensor->data_c(), y_tensor->data_c(), result_tensor->data_c(), data_size);
        break;
      }
      case kNumberTypeFloat16: {
        DivImpl<float16>(x_tensor->data_c(), y_tensor->data_c(), result_tensor->data_c(), data_size);
        break;
      }
      case kNumberTypeFloat32: {
        DivImpl<float>(x_tensor->data_c(), y_tensor->data_c(), result_tensor->data_c(), data_size);
        break;
      }
      case kNumberTypeFloat64: {
        DivImpl<double>(x_tensor->data_c(), y_tensor->data_c(), result_tensor->data_c(), data_size);
        break;
      }
      default: {
        MS_EXCEPTION(TypeError) << "For '" << primitive->name()
                                << "', the supported type is in the list: ['bool', 'int8', 'int16', 'int32', 'int64', "
                                   "'uint8', 'uint16', 'uint32', 'uint64', 'float16', 'float32', 'float64'], but got: "
                                << result_type->ToString() << ".";
      }
    }
    return result_tensor;
  }
};
REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("Div", DivFrontendFuncImpl);

}  // namespace ops
}  // namespace mindspore
