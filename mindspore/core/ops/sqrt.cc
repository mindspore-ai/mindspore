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

#include "ops/sqrt.h"
#include <complex>
#include <vector>
#include <string>
#include <memory>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
template <typename T>
void ImplSqrt(void *origin, void *target, size_t size) {
  MS_EXCEPTION_IF_NULL(origin);
  MS_EXCEPTION_IF_NULL(target);
  auto origin_data = reinterpret_cast<T *>(origin);
  auto target_data = reinterpret_cast<T *>(target);
  for (size_t i = 0; i < size; ++i) {
    if constexpr (std::is_same_v<T, float16>) {
      target_data[i] = sqrt(origin_data[i]);
    } else {
      target_data[i] = std::sqrt(origin_data[i]);
    }
  }
}

abstract::ShapePtr SqrtInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  constexpr int64_t kNumber1 = 1;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, kNumber1, prim_name);
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex0);
  auto x = input_args[kInputIndex0]->BuildShape();
  MS_EXCEPTION_IF_NULL(x);
  auto shape_element = x->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_element);
  return shape_element;
}

TypePtr SqrtInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto x_type = input_args[kInputIndex0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, common_valid_types_with_complex, primitive->name());
  return x_type;
}

ValuePtr SqrtInferValue(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  if (input_args.empty()) {
    return nullptr;
  }
  auto x = input_args[kInputIndex0]->BuildValue();
  if (x == nullptr) {
    return nullptr;
  }
  auto x_tensor = x->cast<tensor::TensorPtr>();
  if (x_tensor == nullptr) {
    return nullptr;
  }

  auto data_size = x_tensor->DataSize();
  auto dtype = x_tensor->data_type();
  auto shape = SqrtInferShape(prim, input_args)->shape();
  // Same shape and dtype
  auto result_tensor = std::make_shared<tensor::Tensor>(dtype, shape);
  auto x_datac = x_tensor->data_c();
  auto result_datac = result_tensor->data_c();
  switch (dtype) {
    case kNumberTypeInt8: {
      ImplSqrt<int8_t>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeInt16: {
      ImplSqrt<int16_t>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeInt32: {
      ImplSqrt<int32_t>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeInt64: {
      ImplSqrt<int64_t>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeUInt8: {
      ImplSqrt<uint8_t>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeUInt16: {
      ImplSqrt<uint16_t>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeUInt32: {
      ImplSqrt<uint32_t>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeUInt64: {
      ImplSqrt<uint64_t>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeFloat16: {
      ImplSqrt<float16>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeFloat32: {
      ImplSqrt<float>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeFloat64: {
      ImplSqrt<double>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeComplex64: {
      ImplSqrt<std::complex<float>>(x_datac, result_datac, data_size);
      break;
    }
    case kNumberTypeComplex128: {
      ImplSqrt<std::complex<double>>(x_datac, result_datac, data_size);
      break;
    }
    default: {
      MS_EXCEPTION(TypeError)
        << "For '" << prim->name()
        << "', the supported data type is ['int8', 'int16', 'int32', 'int64', 'uint8', "
           "'uint16','uint32', 'uint64','float16', 'float32', 'float64', 'complex64', 'complex128'], but got "
        << TypeIdToString(dtype) << ".";
    }
  }
  return result_tensor;
}
}  // namespace

MIND_API_OPERATOR_IMPL(Sqrt, BaseOperator);
AbstractBasePtr SqrtInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  constexpr int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());

  return abstract::MakeAbstract(SqrtInferShape(primitive, input_args), SqrtInferType(primitive, input_args));
}
REGISTER_PRIMITIVE_EVAL_IMPL(Sqrt, prim::kPrimSqrt, SqrtInfer, SqrtInferValue, true);
}  // namespace ops
}  // namespace mindspore
