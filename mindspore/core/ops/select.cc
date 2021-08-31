/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "ops/select.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "utils/tensor_construct_utils.h"
namespace mindspore {
namespace ops {
namespace {
constexpr auto kCondIndex = 0;
constexpr auto kXIndex = 1;
constexpr auto kYIndex = 2;
template <typename T>
void SelectImpl(const bool *conds, void *x, void *y, void *result, size_t size) {
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(y);
  MS_EXCEPTION_IF_NULL(result);
  MS_EXCEPTION_IF_NULL(conds);
  T *x_data = reinterpret_cast<T *>(x);
  T *y_data = reinterpret_cast<T *>(y);
  auto result_data = reinterpret_cast<T *>(result);
  MS_EXCEPTION_IF_NULL(x_data);
  MS_EXCEPTION_IF_NULL(y_data);
  MS_EXCEPTION_IF_NULL(result_data);
  for (size_t i = 0; i < size; ++i) {
    auto cond = conds[i];
    result_data[i] = cond ? x_data[i] : y_data[i];
  }
}
abstract::BaseShapePtr SelectInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto cond_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kCondIndex]->BuildShape());
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kXIndex]->BuildShape());
  auto y_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kYIndex]->BuildShape());
  bool error_flag = false;
  if (x_shape[kShape] != cond_shape[kShape] || x_shape[kShape] != y_shape[kShape]) {
    error_flag = true;
  }
  if (CheckAndConvertUtils::HasDynamicShapeInput(input_args)) {
    if (x_shape[kMaxShape] != cond_shape[kMaxShape] || x_shape[kMaxShape] != y_shape[kMaxShape]) {
      error_flag = true;
    }
    if (x_shape[kMinShape] != cond_shape[kMinShape] || x_shape[kMinShape] != y_shape[kMinShape]) {
      error_flag = true;
    }
  }
  if (error_flag) {
    MS_LOG(ERROR) << " cond shape :" << input_args[kCondIndex]->BuildShape()->ToString();
    MS_LOG(ERROR) << " x shape :" << input_args[kXIndex]->BuildShape()->ToString();
    MS_LOG(ERROR) << " y shape :" << input_args[kYIndex]->BuildShape()->ToString();
    MS_EXCEPTION(ValueError) << "The x_shape is not same as y_shape and cond_shape";
  }
  return input_args[1]->BuildShape();
}

TypePtr SelectInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  auto x_type = input_args[kXIndex]->BuildType();
  auto y_type = input_args[kYIndex]->BuildType();
  auto cond_type = input_args[kCondIndex]->BuildType();
  (void)CheckAndConvertUtils::CheckSubClass("x_type", x_type, {kTensorType}, prim_name);
  (void)CheckAndConvertUtils::CheckSubClass("y_type", y_type, {kTensorType}, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("cond", cond_type, {kBool}, prim_name);
  if (*x_type != *y_type) {
    MS_EXCEPTION(TypeError) << prim_name << "the x_type " << x_type->ToString() << " must be the same as y_type "
                            << y_type->ToString();
  }
  return x_type;
}
AbstractBasePtr SelectInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args) {
  const int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, "ops [select]");
  auto type = SelectInferType(primitive, input_args);
  auto shape = SelectInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}

ValuePtr SelectInferValue(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto result_type = SelectInferType(prim, input_args);
  auto result_shape = SelectInferShape(prim, input_args)->cast<abstract::ShapePtr>();
  auto cond_value = input_args[kCondIndex]->BuildValue();
  auto x = input_args[kXIndex]->BuildValue();
  auto y = input_args[kYIndex]->BuildValue();
  if (x == nullptr || y == nullptr || cond_value == nullptr || result_shape->IsDynamic()) {
    return nullptr;
  }
  auto x_tensor = x->cast<tensor::TensorPtr>();
  auto y_tensor = y->cast<tensor::TensorPtr>();
  auto cond_tensor = cond_value->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  MS_EXCEPTION_IF_NULL(y_tensor);
  MS_EXCEPTION_IF_NULL(cond_tensor);
  auto conds = cond_tensor->data_c();
  MS_EXCEPTION_IF_NULL(conds);
  bool *cond_data = reinterpret_cast<bool *>(cond_tensor->data_c());
  auto data_size = cond_tensor->DataSize();
  auto type_id = x_tensor->data_type();
  auto result_tensor = std::make_shared<tensor::Tensor>(type_id, result_shape->shape());
  switch (type_id) {
    case kNumberTypeBool: {
      SelectImpl<bool>(cond_data, x_tensor->data_c(), y_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeInt: {
      SelectImpl<int>(cond_data, x_tensor->data_c(), y_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeInt8: {
      SelectImpl<int8_t>(cond_data, x_tensor->data_c(), y_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeInt16: {
      SelectImpl<int16_t>(cond_data, x_tensor->data_c(), y_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeInt32: {
      SelectImpl<int32_t>(cond_data, x_tensor->data_c(), y_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeInt64: {
      SelectImpl<int64_t>(cond_data, x_tensor->data_c(), y_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeUInt: {
      SelectImpl<uint32_t>(cond_data, x_tensor->data_c(), y_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeUInt8: {
      SelectImpl<uint8_t>(cond_data, x_tensor->data_c(), y_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeUInt16: {
      SelectImpl<uint16_t>(cond_data, x_tensor->data_c(), y_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeUInt32: {
      SelectImpl<uint32_t>(cond_data, x_tensor->data_c(), y_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeUInt64: {
      SelectImpl<uint64_t>(cond_data, x_tensor->data_c(), y_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeFloat: {
      SelectImpl<float>(cond_data, x_tensor->data_c(), y_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeFloat16: {
      SelectImpl<float16>(cond_data, x_tensor->data_c(), y_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeFloat32: {
      SelectImpl<float>(cond_data, x_tensor->data_c(), y_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeFloat64: {
      SelectImpl<double>(cond_data, x_tensor->data_c(), y_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    default: {
      MS_EXCEPTION(TypeError) << "Select not supported type " << result_type->ToString();
    }
  }
  return result_tensor;
}
}  // namespace
REGISTER_PRIMITIVE_EVAL_IMPL(Select, prim::kPrimSelect, SelectInfer, SelectInferValue, true);
}  // namespace ops
}  // namespace mindspore
