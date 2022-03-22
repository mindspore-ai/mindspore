/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "ops/random_poisson.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include <map>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/param_validator.h"
#include "mindapi/src/helper.h"
namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr RandomPoissonInferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  if (!input_args[0]->isa<abstract::AbstractTensor>()) {
    MS_EXCEPTION(TypeError) << "For RandomPoisson, input[0] only support tensor!";
  }
  MS_EXCEPTION_IF_NULL(primitive);
  const uint32_t kInpuDims = 1;
  auto max_length_ptr = primitive->GetAttr("max_length");
  MS_EXCEPTION_IF_NULL(max_length_ptr);
  int64_t max_length = GetValue<int64_t>(max_length_ptr);
  auto input_shape = input_args[0]->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_shape);
  auto input_shape_value_ptr = input_shape->BuildValue();
  MS_EXCEPTION_IF_NULL(input_shape_value_ptr);
  auto input_shape_tensor = input_shape_value_ptr->cast<tensor::TensorPtr>();
  auto input_type = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(input_type);
  auto input_type_id = input_type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(input_type_id);
  auto input_type_element = input_type_id->element();
  MS_EXCEPTION_IF_NULL(input_type_element);
  auto shape_ptr = std::make_shared<abstract::Shape>(
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape]);
  auto shape_v = shape_ptr->shape();
  if (shape_v.size() != kInpuDims) {
    MS_EXCEPTION(ValueError) << "For RandomPoisson, the input tensor must be a 1-D tensor.";
  }
  if (!input_args[0]->BuildValue()->isa<AnyValue>() && !input_args[0]->BuildValue()->isa<None>()) {
    std::vector<int64_t> out_shape;
    auto shape_m = 1;
    if (input_type_element->type_id() == kNumberTypeInt32) {
      auto input_shape_ptr = reinterpret_cast<int32_t *>(input_shape_tensor->data_c());
      for (auto i = 0; i < shape_v[0]; ++i) {
        if (input_shape_ptr[i] > 0) {
          out_shape.push_back(input_shape_ptr[i]);
          shape_m *= input_shape_ptr[i];
        } else {
          MS_EXCEPTION(ValueError) << "For RandomPoisson, each dimension must be greater than 0.";
        }
      }
    } else if (input_type_element->type_id() == kNumberTypeInt64) {
      auto input_shape_ptr = reinterpret_cast<int64_t *>(input_shape_tensor->data_c());
      for (auto i = 0; i < shape_v[0]; ++i) {
        if (input_shape_ptr[i] > 0) {
          out_shape.push_back(input_shape_ptr[i]);
          shape_m *= input_shape_ptr[i];
        } else {
          MS_EXCEPTION(ValueError) << "For RandomPoisson, each dimension must be greater than 0.";
        }
      }
    }
    if (shape_m > max_length) {
      MS_EXCEPTION(ValueError) << "For RandomPoisson, the number of elements of output must be less than max length: "
                               << max_length << ", but got " << shape_m
                               << "! the shape of  output should be reduced or max_length should be increased";
    }
    return std::make_shared<abstract::Shape>(out_shape);
  } else {
    const uint32_t input_shapes = static_cast<uint32_t>(std::pow(max_length, 1.0 / shape_v[0]));
    std::vector<int64_t> output_shape;
    ShapeVector shape_min;
    ShapeVector shape_max;
    for (int i = 0; i < shape_v[0]; i++) {
      output_shape.push_back(abstract::Shape::SHP_ANY);
      shape_min.push_back(0);
      shape_max.push_back(input_shapes);
    }
    return std::make_shared<abstract::Shape>(output_shape, shape_min, shape_max);
  }
}

TypePtr RandomPoissonInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  const std::set<TypePtr> valid_shape_types = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTypeValid("shape", input_args[0]->BuildType(), valid_shape_types, prim_name);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64, kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTypeValid("rate", input_args[1]->BuildType(), valid_types, prim_name);
  auto dtype_value = prim->GetAttr("dtype");
  if (!dtype_value->isa<Type>()) {
    MS_EXCEPTION(TypeError) << "For RandomPoisson, the dtype of " + prim_name + " is invalid!";
  }
  auto output_type = dtype_value->cast<TypePtr>();
  return CheckAndConvertUtils::CheckSubClass("dtype", output_type, valid_types, prim_name);
}
}  // namespace

AbstractBasePtr RandomPoissonInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infertype = RandomPoissonInferType(primitive, input_args);
  auto infershape = RandomPoissonInferShape(primitive, input_args);
  return abstract::MakeAbstract(infershape, infertype);
}

MIND_API_OPERATOR_IMPL(RandomPoisson, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(RandomPoisson, prim::kPrimRandomPoisson, RandomPoissonInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
