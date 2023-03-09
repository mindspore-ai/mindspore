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

#include "ops/log.h"
#include <string>
#include <vector>
#include <set>
#include <map>
#include <memory>
#include <complex>
#include <cmath>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

void SetDefaultAttrs(const PrimitivePtr &primitive) {
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string kBaseName = "base";
  const std::string kScaleName = "scale";
  const std::string kShiftName = "shift";
  constexpr float value = 1.0;
  if (primitive->GetAttr(kBaseName) == nullptr) {
    primitive->set_attr(kBaseName, MakeValue(value));
  }
  if (primitive->GetAttr(kScaleName) == nullptr) {
    primitive->set_attr(kScaleName, MakeValue(value));
  }
  if (primitive->GetAttr(kShiftName) == nullptr) {
    primitive->set_attr(kShiftName, MakeValue(value));
  }
}

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

abstract::ShapePtr LogInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 0);
  auto x = input_args[kInputIndex0]->BuildShape();
  MS_EXCEPTION_IF_NULL(x);
  auto shape_element = x->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_element);
  return shape_element;
}

TypePtr LogInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto op_name = prim->name();
  (void)CheckAndConvertUtils::CheckInteger("input number", int64_t(input_args.size()), kEqual, 1, op_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64, kComplex64, kComplex128};
  auto x_type = input_args[kInputIndex0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, op_name);
  return x_type;
}
ValuePtr LogInferValue(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
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
  auto infer_shape = LogInferShape(prim, input_args);
  MS_EXCEPTION_IF_NULL(infer_shape);
  auto shape = infer_shape->shape();
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
        << "For '" << prim->name()
        << "', the supported data types are ['float16', 'float32', 'float64', 'complex64', 'complex128'], but got "
        << x_tensor->ToString();
    }
  }
  return result_tensor;
}
}  // namespace

MIND_API_OPERATOR_IMPL(Log, BaseOperator);
AbstractBasePtr LogInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                         const std::vector<AbstractBasePtr> &input_args) {
  SetDefaultAttrs(primitive);
  auto type = LogInferType(primitive, input_args);
  auto shape = LogInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}

// AG means auto generated
class MIND_API AGLogInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return LogInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return LogInferType(primitive, input_args);
  }
  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return LogInferValue(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return LogInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Log, prim::kPrimLog, AGLogInfer, true);
}  // namespace ops
}  // namespace mindspore
