/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "ops/not_equal.h"
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>
#include <complex>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
template <typename T>
void NotEqualImpl(void *x1, void *x2, void *result, size_t size) {
  MS_EXCEPTION_IF_NULL(x1);
  MS_EXCEPTION_IF_NULL(x2);
  MS_EXCEPTION_IF_NULL(result);
  T *x1_data = static_cast<T *>(x1);
  T *x2_data = static_cast<T *>(x2);
  auto result_data = static_cast<bool *>(result);
  for (size_t i = 0; i < size; ++i) {
    result_data[i] = !(x1_data[i] == x2_data[i]);
  }
}

template <typename T>
void NotEqualFloatImpl(void *x1, void *x2, void *result, size_t size) {
  MS_EXCEPTION_IF_NULL(x1);
  MS_EXCEPTION_IF_NULL(x2);
  MS_EXCEPTION_IF_NULL(result);
  T *x1_data = static_cast<T *>(x1);
  T *x2_data = static_cast<T *>(x2);
  auto result_data = static_cast<bool *>(result);
  for (size_t i = 0; i < size; ++i) {
    result_data[i] = std::abs(x1_data[i] - x2_data[i]) > std::numeric_limits<T>::epsilon();
  }
}

abstract::ShapePtr NotEqualInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  return BroadCastInferShape(prim_name, input_args);
}

TypePtr NotEqualInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = prim->name();
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[0]->BuildType());
  (void)types.emplace("y", input_args[1]->BuildType());
  const std::set<TypePtr> valid_types = {kFloat64, kBool,   kInt64,  kFloat,  kFloat16, kInt16,     kInt32,
                                         kInt8,    kUInt16, kUInt32, kUInt64, kUInt8,   kComplex64, kComplex128};
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);
  return std::make_shared<TensorType>(kBool);
}

ValuePtr NotEqualInferValue(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto result_type = NotEqualInferType(prim, input_args);
  auto result_shape = NotEqualInferShape(prim, input_args)->cast<abstract::ShapePtr>();
  constexpr size_t input_size = 2;
  if (input_args.size() != input_size) {
    MS_LOG(ERROR) << "input_args.size is not equal to 2";
  }
  auto x1 = input_args[0]->BuildValue();
  auto x2 = input_args[1]->BuildValue();
  if (x1 == nullptr || x2 == nullptr) {
    return nullptr;
  }
  auto x1_tensor = x1->cast<tensor::TensorPtr>();
  auto x2_tensor = x2->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x1_tensor);
  MS_EXCEPTION_IF_NULL(x2_tensor);
  auto type_id = x1_tensor->data_type();
  auto data_size = x1_tensor->DataSize();
  auto result_tensor = std::make_shared<tensor::Tensor>(kNumberTypeBool, result_shape->shape());
  switch (type_id) {
    case kNumberTypeBool: {
      NotEqualImpl<bool>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeInt: {
      NotEqualImpl<int>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeInt8: {
      NotEqualImpl<int8_t>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeInt16: {
      NotEqualImpl<int16_t>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeInt32: {
      NotEqualImpl<int32_t>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeInt64: {
      NotEqualImpl<int64_t>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeUInt8: {
      NotEqualImpl<uint8_t>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeFloat: {
      NotEqualFloatImpl<float>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeFloat16: {
      NotEqualImpl<float16>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeFloat32: {
      NotEqualFloatImpl<float>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeFloat64: {
      NotEqualFloatImpl<double>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeComplex64: {
      NotEqualImpl<std::complex<float>>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeComplex128: {
      NotEqualImpl<std::complex<double>>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    default: {
      MS_EXCEPTION(TypeError) << "For '" << prim->name()
                              << "', the supported type is in the list: ['bool', 'int8', 'int16', 'int32', 'int64', "
                                 "'complex64', 'complex128', 'uint8', 'float16', 'float32', 'float64'], but got "
                              << result_type->ToString() << ".";
    }
  }
  return result_tensor;
}
}  // namespace

MIND_API_OPERATOR_IMPL(NotEqual, BaseOperator);
AbstractBasePtr NotEqualInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args) {
  constexpr int64_t kInputNum = 2;
  MS_EXCEPTION_IF_NULL(primitive);
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, primitive->name());
  auto infer_type = NotEqualInferType(primitive, input_args);
  auto infer_shape = NotEqualInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(NotEqual, prim::kPrimNotEqual, NotEqualInfer, NotEqualInferValue, true);
}  // namespace ops
}  // namespace mindspore
