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

#include "ops/square.h"
#include "abstract/primitive_infer_map.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
namespace {
template <typename T>
void ImpleSquare(void *origin, void *target, size_t size) {
  MS_EXCEPTION_IF_NULL(origin);
  MS_EXCEPTION_IF_NULL(target);
  auto origin_data = reinterpret_cast<T *>(origin);
  auto target_data = reinterpret_cast<T *>(target);
  for (size_t i = 0; i < size; ++i) {
    target_data[i] = origin_data[i] * origin_data[i];
  }
}

abstract::ShapePtr SquareInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape());
  auto in_shape = shape_map[kShape];
  auto min_shape = shape_map[kMinShape];
  auto max_shape = shape_map[kMaxShape];
  return std::make_shared<abstract::Shape>(in_shape, min_shape, max_shape);
}

TypePtr SquareInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto x_dtype = input_args[kInputIndex0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_dtype, common_valid_types, prim->name());
  return x_dtype;
}

AbstractBasePtr SquareInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());

  return abstract::MakeAbstract(SquareInferShape(primitive, input_args), SquareInferType(primitive, input_args));
}

ValuePtr SquareInferValue(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
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
  auto shape = SquareInferShape(prim, input_args)->shape();
  auto result_tensor = std::make_shared<tensor::Tensor>(dtype, shape);  // same shape and dtype
  auto x_datac = x_tensor->data_c();
  auto result_datac = result_tensor->data_c();
  switch (dtype) {
    case kNumberTypeInt8: {
      ImpleSquare<int8_t>(x_datac, result_datac, IntToSize(data_size));
      break;
    }
    case kNumberTypeInt16: {
      ImpleSquare<int16_t>(x_datac, result_datac, IntToSize(data_size));
      break;
    }
    case kNumberTypeInt32: {
      ImpleSquare<int32_t>(x_datac, result_datac, IntToSize(data_size));
      break;
    }
    case kNumberTypeInt64: {
      ImpleSquare<int64_t>(x_datac, result_datac, IntToSize(data_size));
      break;
    }
    case kNumberTypeUInt8: {
      ImpleSquare<uint8_t>(x_datac, result_datac, IntToSize(data_size));
      break;
    }
    case kNumberTypeUInt16: {
      ImpleSquare<uint16_t>(x_datac, result_datac, IntToSize(data_size));
      break;
    }
    case kNumberTypeUInt32: {
      ImpleSquare<uint32_t>(x_datac, result_datac, IntToSize(data_size));
      break;
    }
    case kNumberTypeUInt64: {
      ImpleSquare<uint64_t>(x_datac, result_datac, IntToSize(data_size));
      break;
    }
    case kNumberTypeFloat16: {
      ImpleSquare<float16>(x_datac, result_datac, IntToSize(data_size));
      break;
    }
    case kNumberTypeFloat32: {
      ImpleSquare<float>(x_datac, result_datac, IntToSize(data_size));
      break;
    }
    case kNumberTypeFloat64: {
      ImpleSquare<double>(x_datac, result_datac, IntToSize(data_size));
      break;
    }
    default: {
      MS_EXCEPTION(TypeError) << "Square unsupported data type: " << x_tensor->ToString();
    }
  }
  return result_tensor;
}
}  // namespace
REGISTER_PRIMITIVE_EVAL_IMPL(Square, prim::kPrimSquare, SquareInfer, SquareInferValue, true);
}  // namespace ops
}  // namespace mindspore
