/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "ops/exp.h"
#include <cmath>
#include <complex>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/math_ops.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

template <typename T>
void ImpleExp(void *origin, void *target, size_t size) {
  MS_EXCEPTION_IF_NULL(origin);
  MS_EXCEPTION_IF_NULL(target);
  auto origin_data = static_cast<T *>(origin);
  auto target_data = static_cast<T *>(target);
  for (size_t i = 0; i < size; ++i) {
    target_data[i] = static_cast<T>(exp(static_cast<double>(origin_data[i])));
  }
}

template <typename T>
void ImpleComplexExp(void *origin, void *target, size_t size) {
  MS_EXCEPTION_IF_NULL(origin);
  MS_EXCEPTION_IF_NULL(target);
  auto origin_data = static_cast<T *>(origin);
  auto target_data = static_cast<T *>(target);
  for (size_t i = 0; i < size; ++i) {
    target_data[i] = static_cast<T>(exp(origin_data[i]));
  }
}

abstract::ShapePtr ExpInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input numbers", int64_t(input_args.size()), kEqual, 1, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto x = input_args[0]->BuildShape();
  MS_EXCEPTION_IF_NULL(x);
  auto shape_ptr = x->cast<abstract::ShapePtr>();
  if (shape_ptr == nullptr) {
    MS_EXCEPTION(TypeError) << "For '" << primitive->name() << "', the input x only support tensor!";
  }
  return shape_ptr;
}

TypePtr ExpInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, 1, prim->name());
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[0]->BuildType());
  std::set<TypePtr> valid_params_types = {kTensorType};
  if (!input_args[0]->isa<abstract::AbstractTensor>()) {
    MS_EXCEPTION(TypeError) << "For '" << prim->name() << "', the input x only support tensor!";
  }
  (void)CheckAndConvertUtils::CheckSubClass("x_type", input_args[0]->BuildType(), valid_params_types, prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, common_valid_types_with_complex_and_bool, prim->name());
  return input_args[0]->BuildType();
}

ValuePtr ExpInferValue(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  if (input_args.empty()) {
    return nullptr;
  }
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto x = input_args[kInputIndex0]->BuildValue();
  if (x == nullptr) {
    return nullptr;
  }
  MS_EXCEPTION_IF_NULL(x);
  auto x_tensor = x->cast<tensor::TensorPtr>();
  if (x_tensor == nullptr) {
    return nullptr;
  }
  MS_EXCEPTION_IF_NULL(x_tensor);
  auto data_size = x_tensor->DataSize();
  auto dtype = x_tensor->data_type();
  auto infer_shape = ExpInferShape(prim, input_args);
  MS_EXCEPTION_IF_NULL(infer_shape);
  auto shape = infer_shape->shape();
  auto result_tensor = std::make_shared<tensor::Tensor>(dtype, shape);  // same shape and dtype
  auto x_datac = x_tensor->data_c();
  MS_EXCEPTION_IF_NULL(result_tensor);
  auto result_datac = result_tensor->data_c();
  switch (dtype) {
    case kNumberTypeInt8: {
      ImpleExp<int8_t>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeInt16: {
      ImpleExp<int16_t>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeInt32: {
      ImpleExp<int32_t>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeInt64: {
      ImpleExp<int64_t>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeUInt8: {
      ImpleExp<uint8_t>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeUInt16: {
      ImpleExp<uint16_t>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeUInt32: {
      ImpleExp<uint32_t>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeUInt64: {
      ImpleExp<uint64_t>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeFloat16: {
      ImpleExp<float16>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeFloat32: {
      ImpleExp<float>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeFloat64: {
      ImpleExp<double>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeComplex64: {
      ImpleComplexExp<std::complex<float>>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeComplex128: {
      ImpleComplexExp<std::complex<double>>(x_datac, result_datac, data_size);
      break;
    }
    default: {
      MS_EXCEPTION(TypeError)
        << "For '" << prim->name()
        << "', the supported data type is ['int8', 'int16', 'int32', 'int64', 'uint8', "
           "'uint16','uint32', 'uint64','float16', 'float32', 'float64', 'complex64', 'complex128'], but got "
        << x_tensor->ToString();
    }
  }
  return result_tensor;
}
}  // namespace

MIND_API_OPERATOR_IMPL(Exp, BaseOperator);
AbstractBasePtr ExpInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                         const std::vector<AbstractBasePtr> &input_args) {
  return abstract::MakeAbstract(ExpInferShape(primitive, input_args), ExpInferType(primitive, input_args));
}

// AG means auto generated
class MIND_API AGExpInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ExpInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ExpInferType(primitive, input_args);
  }
  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ExpInferValue(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ExpInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Exp, prim::kPrimExp, AGExpInfer, true);
}  // namespace ops
}  // namespace mindspore
