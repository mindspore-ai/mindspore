/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#include "ops/eps.h"
#include <memory>
#include <vector>
#include <limits>
#include <set>
#include <cmath>
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"
#include "base/float16.h"

namespace mindspore {
namespace ops {
namespace {
template <typename T>
T getEpsilon() {
  T epsilon = static_cast<T>(0.5);
  T one = static_cast<T>(1.0);
  T two = static_cast<T>(2.0);
  while (one + epsilon / two > one) {
    epsilon = epsilon / two;
  }
  return epsilon;
}

template <typename T>
void ImpleEps(void *target, size_t size) {
  T min_val = getEpsilon<T>();
  auto target_data = static_cast<T *>(target);
  for (size_t i = 0; i < size; ++i) {
    target_data[i] = min_val;
  }
}

abstract::ShapePtr EpsInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  return CheckAndConvertUtils::GetTensorInputShape(primitive->name(), input_args, 0);
}

TypePtr EpsInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto infer_type = input_args[0]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", infer_type, valid_types, primitive->name());
  return infer_type;
}

ValuePtr EpsInferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  if (input_args.empty()) {
    return nullptr;
  }
  auto x = input_args[0]->BuildValue();
  if (x == nullptr) {
    return nullptr;
  }
  auto x_tensor = x->cast<tensor::TensorPtr>();
  if (x_tensor == nullptr) {
    return nullptr;
  }
  auto data_size = x_tensor->DataSize();
  auto dtype = x_tensor->data_type();
  auto shape = EpsInferShape(primitive, input_args)->cast<abstract::ShapePtr>();
  auto result_tensor = std::make_shared<tensor::Tensor>(dtype, shape->shape());
  auto result_datac = result_tensor->data_c();
  switch (dtype) {
    case kNumberTypeFloat16: {
      ImpleEps<float16>(result_datac, data_size);
      break;
    }
    case kNumberTypeFloat32: {
      ImpleEps<float>(result_datac, data_size);
      break;
    }
    case kNumberTypeFloat64: {
      ImpleEps<double>(result_datac, data_size);
      break;
    }
    default: {
      MS_EXCEPTION(TypeError) << "For '" << primitive->name()
                              << "', the supported data types are ['float16', 'float32', 'float64'], but got: "
                              << x_tensor->ToString() << ".";
    }
  }
  return result_tensor;
}
}  // namespace

AbstractBasePtr EpsInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = EpsInferType(primitive, input_args);
  auto infer_shape = EpsInferShape(primitive, input_args);
  return abstract::MakeAbstractTensor(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(Eps, BaseOperator);
class MIND_API AGEpsInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return EpsInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return EpsInferType(primitive, input_args);
  }

  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return EpsInferValue(primitive, input_args);
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return EpsInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Eps, prim::kPrimEps, AGEpsInfer, true);
}  // namespace ops
}  // namespace mindspore
