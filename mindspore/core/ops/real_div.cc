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

#include "ops/real_div.h"
#include <map>
#include <memory>
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
abstract::ShapePtr RealDivInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  return BroadCastInferShape(prim_name, input_args);
}

TypePtr RealDivInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto op_name = prim->name();
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("infer", SizeToLong(input_args.size()), kGreaterEqual, input_num, op_name);
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[0]->BuildType());
  (void)types.emplace("y", input_args[1]->BuildType());
  return CheckAndConvertUtils::CheckMathBinaryOpTensorType(types, common_valid_types, op_name);
}
}  // namespace

MIND_API_OPERATOR_IMPL(RealDiv, BaseOperator);
AbstractBasePtr RealDivInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  return abstract::MakeAbstract(RealDivInferShape(primitive, input_args), RealDivInferType(primitive, input_args));
}

// AG means auto generated
class MIND_API AGRealDivInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return RealDivInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return RealDivInferType(primitive, input_args);
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return RealDivInfer(engine, primitive, input_args);
  }

  ValuePtr InferValue(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    const int64_t input_num = 2;
    MS_EXCEPTION_IF_NULL(prim);
    auto op_name = prim->name();
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num, op_name);
    auto x = input_args[0]->BuildValue();
    auto y = input_args[1]->BuildValue();

    if (!IsValueKnown(x) || !IsValueKnown(y) || !input_args[0]->isa<abstract::AbstractTensor>() ||
        !input_args[1]->isa<abstract::AbstractTensor>()) {
      return nullptr;
    }
    auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->GetShapeTrack());
    auto y_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->GetShapeTrack());
    auto x_shape = x_shape_map[kShape];
    auto y_shape = y_shape_map[kShape];
    if (!x_shape.empty() || x_shape != y_shape) {
      return nullptr;
    }
    auto x_tensor = x->cast<tensor::TensorPtr>();
    auto y_tensor = y->cast<tensor::TensorPtr>();
    TypeId x_dtype = x_tensor->data_type();
    TypeId y_dtype = y_tensor->data_type();
    if (x_dtype != y_dtype) {
      return nullptr;
    }
    auto x_datac = x_tensor->data_c();
    auto y_datac = y_tensor->data_c();
    auto result_tensor = std::make_shared<tensor::Tensor>(x_dtype, x_shape);
    auto result_datac = result_tensor->data_c();

    switch (x_dtype) {
      case kNumberTypeInt8: {
        ImpleRealDiv<int8_t>(x_datac, y_datac, result_datac);
        break;
      }
      case kNumberTypeInt16: {
        ImpleRealDiv<int16_t>(x_datac, y_datac, result_datac);
        break;
      }
      case kNumberTypeInt32: {
        ImpleRealDiv<int32_t>(x_datac, y_datac, result_datac);
        break;
      }
      case kNumberTypeInt64: {
        ImpleRealDiv<int64_t>(x_datac, y_datac, result_datac);
        break;
      }
      case kNumberTypeUInt8: {
        ImpleRealDiv<uint8_t>(x_datac, y_datac, result_datac);
        break;
      }
      case kNumberTypeUInt16: {
        ImpleRealDiv<uint16_t>(x_datac, y_datac, result_datac);
        break;
      }
      case kNumberTypeUInt32: {
        ImpleRealDiv<uint32_t>(x_datac, y_datac, result_datac);
        break;
      }
      case kNumberTypeUInt64: {
        ImpleRealDiv<uint64_t>(x_datac, y_datac, result_datac);
        break;
      }
      case kNumberTypeFloat16: {
        ImpleRealDiv<float16>(x_datac, y_datac, result_datac);
        break;
      }
      case kNumberTypeFloat32: {
        ImpleRealDiv<float>(x_datac, y_datac, result_datac);
        break;
      }
      case kNumberTypeFloat64: {
        ImpleRealDiv<double>(x_datac, y_datac, result_datac);
        break;
      }
      default: {
        return nullptr;
      }
    }
    return result_tensor;
  }

 private:
  template <typename T>
  void ImpleRealDiv(void *x, void *y, void *output) const {
    MS_EXCEPTION_IF_NULL(x);
    MS_EXCEPTION_IF_NULL(y);
    auto dividend = reinterpret_cast<T *>(x);
    auto divisor = reinterpret_cast<T *>(y);
    auto out = reinterpret_cast<T *>(output);
    auto zero = static_cast<T>(0);
    if (*divisor == zero) {
      if (*dividend == zero) {
        *out = std::numeric_limits<T>::quiet_NaN();
      }
      if (std::numeric_limits<T>::has_infinity) {
        *out = *dividend > zero ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
      } else {
        *out = *dividend > zero ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
      }
    } else {
      *out = static_cast<T>(*dividend / *divisor);
    }
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(RealDiv, prim::kPrimRealDiv, AGRealDivInfer, true);
}  // namespace ops
}  // namespace mindspore
