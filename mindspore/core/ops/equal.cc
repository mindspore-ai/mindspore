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

#include "ops/equal.h"
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
void EqualImpl(void *x1, void *x2, void *result, size_t size, bool need_broad_cast) {
  MS_EXCEPTION_IF_NULL(x1);
  MS_EXCEPTION_IF_NULL(x2);
  MS_EXCEPTION_IF_NULL(result);
  T *x1_data = static_cast<T *>(x1);
  T *x2_data = static_cast<T *>(x2);
  auto result_data = static_cast<bool *>(result);
  for (size_t i = 0; i < size; ++i) {
    if (need_broad_cast) {
      result_data[i] = x1_data[i] == x2_data[0];
    } else {
      result_data[i] = x1_data[i] == x2_data[i];
    }
  }
}

template <typename T>
void EqualFloatImpl(void *x1, void *x2, void *result, size_t size) {
  MS_EXCEPTION_IF_NULL(x1);
  MS_EXCEPTION_IF_NULL(x2);
  MS_EXCEPTION_IF_NULL(result);
  T *x1_data = static_cast<T *>(x1);
  T *x2_data = static_cast<T *>(x2);
  auto result_data = static_cast<bool *>(result);
  for (size_t i = 0; i < size; ++i) {
    result_data[i] = std::abs(x1_data[i] - x2_data[i]) < std::numeric_limits<T>::epsilon();
  }
}

abstract::ShapePtr EqualInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  return BroadCastInferShape(op_name, input_args);
}

TypePtr EqualInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto x = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim->name(), input_args, 0);
  auto y = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim->name(), input_args, 1);
  (void)abstract::CheckDtypeSame(prim->name(), x, y);
  const std::set<TypePtr> valid_types = {kInt8,    kInt16, kInt32, kInt64,     kFloat,      kFloat16, kUInt16,
                                         kFloat64, kUInt8, kBool,  kComplex64, kComplex128, kUInt32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[0]->BuildType(), valid_types, prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("y", input_args[1]->BuildType(), valid_types, prim->name());
  return std::make_shared<TensorType>(kBool);
}

ValuePtr EqualInferValue(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  constexpr auto kX1Index = 0;
  constexpr auto kX2Index = 1;
  auto result_type = EqualInferType(prim, input_args);
  auto result_shape = EqualInferShape(prim, input_args)->cast<abstract::ShapePtr>();
  auto x1 = input_args[kX1Index]->BuildValue();
  auto x2 = input_args[kX2Index]->BuildValue();
  if (x1 == nullptr || x2 == nullptr) {
    return nullptr;
  }
  auto x1_tensor = x1->cast<tensor::TensorPtr>();
  auto x2_tensor = x2->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x1_tensor);
  MS_EXCEPTION_IF_NULL(x2_tensor);
  auto type_id = x1_tensor->data_type();
  auto data_size = x1_tensor->DataSize();
  bool need_broad_cast = false;
  if (x1_tensor->DataSize() != x2_tensor->DataSize() && x2_tensor->DataSize() == 1) {
    need_broad_cast = true;
  }
  auto result_tensor = std::make_shared<tensor::Tensor>(kNumberTypeBool, result_shape->shape());
  switch (type_id) {
    case kNumberTypeBool: {
      EqualImpl<bool>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size, need_broad_cast);
      break;
    }
    case kNumberTypeInt: {
      EqualImpl<int>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size, need_broad_cast);
      break;
    }
    case kNumberTypeInt8: {
      EqualImpl<int8_t>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size, need_broad_cast);
      break;
    }
    case kNumberTypeInt16: {
      EqualImpl<int16_t>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size, need_broad_cast);
      break;
    }
    case kNumberTypeInt32: {
      EqualImpl<int32_t>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size, need_broad_cast);
      break;
    }
    case kNumberTypeInt64: {
      EqualImpl<int64_t>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size, need_broad_cast);
      break;
    }
    case kNumberTypeUInt8: {
      EqualImpl<uint8_t>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size, need_broad_cast);
      break;
    }
    case kNumberTypeFloat: {
      EqualFloatImpl<float>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeFloat16: {
      EqualImpl<float16>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size, need_broad_cast);
      break;
    }
    case kNumberTypeFloat32: {
      EqualFloatImpl<float>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeFloat64: {
      EqualFloatImpl<double>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeComplex64: {
      EqualImpl<std::complex<float>>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size,
                                     need_broad_cast);
      break;
    }
    case kNumberTypeComplex128: {
      EqualImpl<std::complex<double>>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size,
                                      need_broad_cast);
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

MIND_API_OPERATOR_IMPL(Equal, BaseOperator);
AbstractBasePtr EqualInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  auto shape = EqualInferShape(primitive, input_args);
  auto type = EqualInferType(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}

// AG means auto generated
class MIND_API AGEqualInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return EqualInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return EqualInferType(primitive, input_args);
  }
  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return EqualInferValue(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return EqualInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Equal, prim::kPrimEqual, AGEqualInfer, true);
}  // namespace ops
}  // namespace mindspore
