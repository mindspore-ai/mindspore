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
ShapeVector GetOffsetVec(ShapeVector shape) {
  ShapeVector offsets;
  for (size_t i = 0; i < shape.size(); i++) {
    size_t offset = 1;
    for (size_t j = i + 1; j < shape.size(); j++) {
      offset *= shape[j];
    }
    offsets.push_back(offset);
  }
  return offsets;
}

ShapeVector GetIndexVec(ShapeVector offsets, size_t index) {
  ShapeVector res;
  for (size_t i = 0; i < offsets.size(); i++) {
    if (offsets[i] == 0) {
      return {};
    }
    res.push_back(index / offsets[i]);
    index %= offsets[i];
  }
  return res;
}

size_t GetIndex(ShapeVector shape, ShapeVector offsets, ShapeVector index_vec) {
  size_t res = 0;
  for (size_t i = 0; i < index_vec.size(); i++) {
    if (index_vec[i] < shape[i]) {
      res += offsets[i] * index_vec[i];
    }
  }
  return res;
}

template <typename T>
void EqualImpl(void *x1, void *x2, void *result, ShapeVector x1_shape, ShapeVector x2_shape, ShapeVector y_shape,
               bool need_broad_cast) {
  MS_EXCEPTION_IF_NULL(x1);
  MS_EXCEPTION_IF_NULL(x2);
  MS_EXCEPTION_IF_NULL(result);
  T *x1_data = static_cast<T *>(x1);
  T *x2_data = static_cast<T *>(x2);
  auto x1_offsets = GetOffsetVec(x1_shape);
  auto x2_offsets = GetOffsetVec(x2_shape);
  auto y_offsets = GetOffsetVec(y_shape);
  if (x1_offsets.size() != x2_offsets.size() || x1_offsets.size() != y_offsets.size()) {
    MS_EXCEPTION(ValueError) << "shape is not match, x1_offsets: " << x1_offsets << " , x2_offsets: " << x2_offsets
                             << " , y_offsets: " << y_offsets;
  }
  size_t data_size = std::accumulate(y_shape.begin(), y_shape.end(), 1, std::multiplies<size_t>());
  auto result_data = static_cast<bool *>(result);
  for (size_t i = 0; i < data_size; ++i) {
    if (need_broad_cast) {
      auto y_index_vec = GetIndexVec(y_offsets, i);
      auto x1_index = GetIndex(x1_shape, x1_offsets, y_index_vec);
      auto x2_index = GetIndex(x2_shape, x2_offsets, y_index_vec);
      result_data[i] = x1_data[x1_index] == x2_data[x2_index];
    } else {
      result_data[i] = x1_data[i] == x2_data[i];
    }
  }
}

template <typename T>
void EqualFloatImpl(void *x1, void *x2, void *result, ShapeVector x1_shape, ShapeVector x2_shape, ShapeVector y_shape,
                    bool need_broad_cast) {
  MS_EXCEPTION_IF_NULL(x1);
  MS_EXCEPTION_IF_NULL(x2);
  MS_EXCEPTION_IF_NULL(result);
  T *x1_data = static_cast<T *>(x1);
  T *x2_data = static_cast<T *>(x2);
  auto result_data = static_cast<bool *>(result);
  auto x1_offsets = GetOffsetVec(x1_shape);
  auto x2_offsets = GetOffsetVec(x2_shape);
  auto y_offsets = GetOffsetVec(y_shape);
  if (x1_offsets.size() != x2_offsets.size() || x1_offsets.size() != y_offsets.size()) {
    MS_EXCEPTION(ValueError) << "shape is not match, x1_offsets: " << x1_offsets << " , x2_offsets: " << x2_offsets
                             << " , y_offsets: " << y_offsets;
  }
  size_t data_size = std::accumulate(y_shape.begin(), y_shape.end(), 1, std::multiplies<size_t>());
  for (size_t i = 0; i < data_size; ++i) {
    if (need_broad_cast) {
      auto y_index_vec = GetIndexVec(y_offsets, i);
      auto x1_index = GetIndex(x1_shape, x1_offsets, y_index_vec);
      auto x2_index = GetIndex(x2_shape, x2_offsets, y_index_vec);
      result_data[i] = std::abs(x1_data[x1_index] - x2_data[x2_index]) < std::numeric_limits<T>::epsilon();
    } else {
      result_data[i] = std::abs(x1_data[i] - x2_data[i]) < std::numeric_limits<T>::epsilon();
    }
  }
}

bool IsBroadCast(ShapeVector x1_shape, ShapeVector x2_shape, ShapeVector *broad_cast_x1_shape,
                 ShapeVector *broad_cast_x2_shape) {
  MS_EXCEPTION_IF_NULL(broad_cast_x1_shape);
  MS_EXCEPTION_IF_NULL(broad_cast_x2_shape);
  bool need_broad_cast = false;
  if (x1_shape.size() != x2_shape.size()) {
    need_broad_cast = true;
  }
  size_t max_size = x1_shape.size() > x2_shape.size() ? x1_shape.size() : x2_shape.size();
  for (size_t i = 0; i < max_size - x1_shape.size(); i++) {
    broad_cast_x1_shape->insert(broad_cast_x1_shape->begin(), 1);
  }
  for (size_t i = 0; i < max_size - x2_shape.size(); i++) {
    broad_cast_x2_shape->insert(broad_cast_x2_shape->begin(), 1);
  }
  for (size_t i = 0; i < max_size; i++) {
    if (broad_cast_x1_shape[i] != broad_cast_x2_shape[i] && broad_cast_x1_shape->at(i) != 1 &&
        broad_cast_x2_shape->at(i) != 1) {
      MS_EXCEPTION(NotSupportError) << "input shape is not match, x1 shape: " << x1_shape
                                    << " , x2 shape: " << x2_shape;
    }
    if (broad_cast_x1_shape[i] != broad_cast_x2_shape[i]) {
      need_broad_cast = true;
    }
  }
  return need_broad_cast;
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
  auto x1_shape = x1_tensor->shape();
  auto x2_shape = x2_tensor->shape();
  ShapeVector broad_cast_x1_shape = x1_shape;
  ShapeVector broad_cast_x2_shape = x2_shape;
  bool need_broad_cast = IsBroadCast(x1_shape, x2_shape, &broad_cast_x1_shape, &broad_cast_x2_shape);
  auto result_tensor = std::make_shared<tensor::Tensor>(kNumberTypeBool, result_shape->shape());
  switch (type_id) {
    case kNumberTypeBool: {
      EqualImpl<bool>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), broad_cast_x1_shape,
                      broad_cast_x2_shape, result_shape->shape(), need_broad_cast);
      break;
    }
    case kNumberTypeInt: {
      EqualImpl<int>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), broad_cast_x1_shape,
                     broad_cast_x2_shape, result_shape->shape(), need_broad_cast);
      break;
    }
    case kNumberTypeInt8: {
      EqualImpl<int8_t>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), broad_cast_x1_shape,
                        broad_cast_x2_shape, result_shape->shape(), need_broad_cast);
      break;
    }
    case kNumberTypeInt16: {
      EqualImpl<int16_t>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), broad_cast_x1_shape,
                         broad_cast_x2_shape, result_shape->shape(), need_broad_cast);
      break;
    }
    case kNumberTypeInt32: {
      EqualImpl<int32_t>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), broad_cast_x1_shape,
                         broad_cast_x2_shape, result_shape->shape(), need_broad_cast);
      break;
    }
    case kNumberTypeInt64: {
      EqualImpl<int64_t>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), broad_cast_x1_shape,
                         broad_cast_x2_shape, result_shape->shape(), need_broad_cast);
      break;
    }
    case kNumberTypeUInt8: {
      EqualImpl<uint8_t>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), broad_cast_x1_shape,
                         broad_cast_x2_shape, result_shape->shape(), need_broad_cast);
      break;
    }
    case kNumberTypeFloat: {
      EqualFloatImpl<float>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), broad_cast_x1_shape,
                            broad_cast_x2_shape, result_shape->shape(), need_broad_cast);
      break;
    }
    case kNumberTypeFloat16: {
      EqualImpl<float16>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), broad_cast_x1_shape,
                         broad_cast_x2_shape, result_shape->shape(), need_broad_cast);
      break;
    }
    case kNumberTypeFloat32: {
      EqualFloatImpl<float>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), broad_cast_x1_shape,
                            broad_cast_x2_shape, result_shape->shape(), need_broad_cast);
      break;
    }
    case kNumberTypeFloat64: {
      EqualFloatImpl<double>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), broad_cast_x1_shape,
                             broad_cast_x2_shape, result_shape->shape(), need_broad_cast);
      break;
    }
    case kNumberTypeComplex64: {
      EqualImpl<std::complex<float>>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(),
                                     broad_cast_x1_shape, broad_cast_x2_shape, result_shape->shape(), need_broad_cast);
      break;
    }
    case kNumberTypeComplex128: {
      EqualImpl<std::complex<double>>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(),
                                      broad_cast_x1_shape, broad_cast_x2_shape, result_shape->shape(), need_broad_cast);
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
