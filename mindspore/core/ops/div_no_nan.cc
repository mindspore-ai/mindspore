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

#include "ops/div_no_nan.h"

#include <map>
#include <set>
#include <string>

#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
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
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
template <typename T>
void DivNoNanImpl(void *x1, void *x2, void *result, size_t size) {
  MS_EXCEPTION_IF_NULL(x1);
  MS_EXCEPTION_IF_NULL(x2);
  MS_EXCEPTION_IF_NULL(result);
  T *x1_data = static_cast<T *>(x1);
  T *x2_data = static_cast<T *>(x2);
  auto result_data = static_cast<T *>(result);
  MS_EXCEPTION_IF_NULL(x1_data);
  MS_EXCEPTION_IF_NULL(x2_data);
  MS_EXCEPTION_IF_NULL(result_data);
  for (size_t i = 0; i < size; ++i) {
    if (x2_data[i] == static_cast<T>(0)) {
      result_data[i] = static_cast<T>(0);
    } else {
      result_data[i] = x1_data[i] / x2_data[i];
    }
  }
}

abstract::ShapePtr DivNoNanInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 0);
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 1);
  return BroadCastInferShape(prim_name, input_args);
}

TypePtr DivNoNanInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  std::map<std::string, TypePtr> types;
  const std::set<TypePtr> valid_types = {kInt8,    kInt32,   kInt64,     kUInt8,     kFloat16,
                                         kFloat32, kFloat64, kComplex64, kComplex128};
  (void)types.emplace("x1", input_args[0]->BuildType());
  (void)types.emplace("x2", input_args[1]->BuildType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
  return input_args[0]->BuildType();
}

ValuePtr DivNoNanInferValue(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  constexpr auto kX1Index = 0;
  constexpr auto kX2Index = 1;
  auto result_type = DivNoNanInferType(prim, input_args);
  auto result_shape = DivNoNanInferShape(prim, input_args)->cast<abstract::ShapePtr>();
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
  auto result_tensor = std::make_shared<tensor::Tensor>(type_id, result_shape->shape());
  switch (type_id) {
    case kNumberTypeBool: {
      DivNoNanImpl<bool>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeInt: {
      DivNoNanImpl<int>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeInt8: {
      DivNoNanImpl<int8_t>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeInt16: {
      DivNoNanImpl<int16_t>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeInt32: {
      DivNoNanImpl<int32_t>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeInt64: {
      DivNoNanImpl<int64_t>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeUInt: {
      DivNoNanImpl<uint32_t>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeUInt8: {
      DivNoNanImpl<uint8_t>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeUInt16: {
      DivNoNanImpl<uint16_t>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeUInt32: {
      DivNoNanImpl<uint32_t>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeUInt64: {
      DivNoNanImpl<uint64_t>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeFloat: {
      DivNoNanImpl<float>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeFloat16: {
      DivNoNanImpl<float16>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeFloat32: {
      DivNoNanImpl<float>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    case kNumberTypeFloat64: {
      DivNoNanImpl<double>(x1_tensor->data_c(), x2_tensor->data_c(), result_tensor->data_c(), data_size);
      break;
    }
    default: {
      MS_EXCEPTION(TypeError) << "For '" << prim->name()
                              << "', the supported type is in the list: ['bool', 'int8', 'int16', 'int32', 'int64', "
                                 "'uint8', 'uint16', 'uint32', 'uint64', 'float16', 'float32', 'float64'], but got: "
                              << result_type->ToString() << ".";
    }
  }
  return result_tensor;
}
}  // namespace

MIND_API_OPERATOR_IMPL(DivNoNan, BaseOperator);
AbstractBasePtr DivNoNanInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infer_type = DivNoNanInferType(primitive, input_args);
  auto infer_shape = DivNoNanInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGDivNoNanInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return DivNoNanInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return DivNoNanInferType(primitive, input_args);
  }
  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return DivNoNanInferValue(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return DivNoNanInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(DivNoNan, prim::kPrimDivNoNan, AGDivNoNanInfer, true);
}  // namespace ops
}  // namespace mindspore
