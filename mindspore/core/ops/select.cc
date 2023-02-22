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

#include <complex>
#include <memory>
#include <vector>

#include "ops/select.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "base/float16.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ir/tensor.h"
#include "mindapi/base/type_id.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
using float_complex = std::complex<float>;
using double_complex = std::complex<double>;
constexpr auto kSelectCondIndex = 0;
constexpr auto kSelectXIndex = 1;
constexpr auto kSelectYIndex = 2;
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

void SelectInferShapeCheck(const std::vector<int64_t> &x_shape, const std::vector<int64_t> &y_shape,
                           const std::vector<int64_t> &cond_shape, size_t shape_size) {
  for (size_t i = 0; i < shape_size; i++) {
    if ((x_shape[i] > 0 && cond_shape[i] > 0 && x_shape[i] != cond_shape[i]) ||
        (x_shape[i] > 0 && y_shape[i] > 0 && x_shape[i] != y_shape[i])) {
      MS_EXCEPTION(ValueError)
        << "For 'Select', the shape of 'condition', 'x' and 'y' must be the same. But got 'condition' shape: "
        << cond_shape << ", 'x' shape: " << x_shape << ", 'y' shape: " << y_shape << ".";
    }
  }
}

abstract::BaseShapePtr SelectInferShape(const PrimitivePtr &, const std::vector<AbstractBasePtr> &input_args) {
  auto cond_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kSelectCondIndex]->BuildShape())[kShape];
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kSelectXIndex]->BuildShape())[kShape];
  auto y_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kSelectYIndex]->BuildShape())[kShape];
  if (IsDynamicRank(cond_shape) || IsDynamicRank(x_shape) || IsDynamicRank(y_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
  }
  auto cond_shape_size = cond_shape.size();
  auto x_shape_size = x_shape.size();
  auto y_shape_size = y_shape.size();
  if (cond_shape_size != x_shape_size || y_shape_size != x_shape_size) {
    MS_EXCEPTION(ValueError)
      << "For 'Select', the shape of 'condition', 'x' and 'y' must be the same. But got 'condition' shape: "
      << cond_shape << ", 'x' shape: " << x_shape << ", 'y' shape: " << y_shape << ".";
  }
  SelectInferShapeCheck(x_shape, y_shape, cond_shape, x_shape_size);
  return input_args[kSelectCondIndex]->BuildShape();
}

TypePtr SelectInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto x_type = input_args[kSelectXIndex]->BuildType();
  auto y_type = input_args[kSelectYIndex]->BuildType();
  auto cond_type = input_args[kSelectCondIndex]->BuildType();
  MS_EXCEPTION_IF_NULL(x_type);
  MS_EXCEPTION_IF_NULL(y_type);

  (void)CheckAndConvertUtils::CheckTensorTypeValid("x_type", x_type, common_valid_types_with_complex_and_bool,
                                                   prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("y_type", y_type, common_valid_types_with_complex_and_bool,
                                                   prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("cond", cond_type, {kBool}, prim_name);
  if (*x_type != *y_type) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name
                            << "', the x_type and y_type must be the same, but got x_type: " << x_type->ToString()
                            << " and y_type: " << y_type->ToString() << ".";
  }
  return x_type;
}

AbstractBasePtr SelectInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, "ops [select]");
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto type = SelectInferType(primitive, input_args);
  auto shape = SelectInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}

void SelectInnerInferValue(const PrimitivePtr &prim, const tensor::TensorPtr &cond_tensor,
                           const tensor::TensorPtr &x_tensor, const tensor::TensorPtr &y_tensor,
                           const tensor::TensorPtr &result_tensor) {
  bool *cond_data = reinterpret_cast<bool *>(cond_tensor->data_c());
  auto data_size = cond_tensor->DataSize();
  auto type_id = x_tensor->data_type();
  switch (type_id) {
    case kNumberTypeBool: {
      SelectImpl<bool>(cond_data, x_tensor->data_c(), y_tensor->data_c(), result_tensor->data_c(), data_size);
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
    case kNumberTypeComplex64: {
      SelectImpl<float_complex>(cond_data, x_tensor->data_c(), y_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeComplex128: {
      SelectImpl<double_complex>(cond_data, x_tensor->data_c(), y_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    default: {
      auto result_type = result_tensor->type();
      MS_EXCEPTION_IF_NULL(result_type);
      MS_EXCEPTION(TypeError) << "For '" << prim->name()
                              << "', the supported data type is ['bool', 'int8', 'int16', 'int32', 'int64', 'uint8', "
                                 "'uint16','uint32', 'uint64','float16', 'float32', 'float64', 'complex64', "
                                 "'complex128'], but got "
                              << result_type->ToString() << ".";
    }
  }
}

ValuePtr SelectInferValue(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  (void)SelectInferType(prim, input_args);
  auto select_shape = SelectInferShape(prim, input_args);
  MS_EXCEPTION_IF_NULL(select_shape);
  auto result_shape = select_shape->cast<abstract::ShapePtr>();
  auto cond_value = input_args[kSelectCondIndex]->BuildValue();
  auto x = input_args[kSelectXIndex]->BuildValue();
  auto y = input_args[kSelectYIndex]->BuildValue();
  MS_EXCEPTION_IF_NULL(result_shape);
  if (x == nullptr || y == nullptr || cond_value == nullptr || result_shape->IsDynamic()) {
    return nullptr;
  }
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(y);
  auto x_tensor = x->cast<tensor::TensorPtr>();
  auto y_tensor = y->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(cond_value);
  auto cond_tensor = cond_value->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  MS_EXCEPTION_IF_NULL(y_tensor);
  MS_EXCEPTION_IF_NULL(cond_tensor);
  auto conds = cond_tensor->data_c();
  MS_EXCEPTION_IF_NULL(conds);
  auto type_id = x_tensor->data_type();
  auto result_tensor = std::make_shared<tensor::Tensor>(type_id, result_shape->shape());
  MS_EXCEPTION_IF_NULL(result_tensor);
  SelectInnerInferValue(prim, cond_tensor, x_tensor, y_tensor, result_tensor);
  return result_tensor;
}
}  // namespace

MIND_API_OPERATOR_IMPL(Select, BaseOperator);

// AG means auto generated
class MIND_API AGSelectInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SelectInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SelectInferType(primitive, input_args);
  }
  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SelectInferValue(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SelectInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Select, prim::kPrimSelect, AGSelectInfer, true);
}  // namespace ops
}  // namespace mindspore
