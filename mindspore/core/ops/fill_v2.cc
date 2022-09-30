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

#include "ops/fill_v2.h"

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "abstract/abstract_value.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(FillV2, BaseOperator);
abstract::ShapePtr FillV2InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  if (!input_args[0]->isa<abstract::AbstractTensor>()) {
    MS_EXCEPTION(TypeError) << "For '" << primitive->name() << "', input[0] must be tensor.";
  }
  MS_EXCEPTION_IF_NULL(primitive);
  const uint32_t kInputDims = 1;
  auto max_length_ptr = primitive->GetAttr("max_length");
  MS_EXCEPTION_IF_NULL(max_length_ptr);
  int64_t max_length = GetValue<int64_t>(max_length_ptr);
  auto input1_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  if (input1_shape.size() != 1) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', the shape size of 'input1' must be 1, but got: " << input1_shape.size() << ".";
  }
  auto input2_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  if (input2_shape.size() != 0) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', the shape size of 'input2' must be 0, but got: " << input2_shape.size() << ".";
  }
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
  if (shape_v.size() != kInputDims) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', input must be a 1-D tensor, but got a: " << shape_v.size() << "-D tensor.";
  }
  if (!input_args[0]->BuildValue()->isa<AnyValue>() && !input_args[0]->BuildValue()->isa<None>()) {
    std::vector<int64_t> out_shape;
    int64_t shape_m = 1;
    if (input_type_element->type_id() == kNumberTypeInt32) {
      auto input_shape_ptr = reinterpret_cast<int32_t *>(input_shape_tensor->data_c());
      for (auto i = 0; i < shape_v[0]; ++i) {
        if (input_shape_ptr[i] > 0) {
          out_shape.push_back(input_shape_ptr[i]);
          shape_m *= input_shape_ptr[i];
        } else {
          MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                                   << "', each dimension of input shape must be greater than 0, but got input shape "
                                   << i << ": " << input_shape_ptr[i] << ".";
        }
      }
    } else if (input_type_element->type_id() == kNumberTypeInt64) {
      auto input_shape_ptr = reinterpret_cast<int64_t *>(input_shape_tensor->data_c());
      for (auto i = 0; i < shape_v[0]; ++i) {
        if (input_shape_ptr[i] > 0) {
          out_shape.push_back(input_shape_ptr[i]);
          shape_m *= static_cast<int64_t>(input_shape_ptr[i]);
        } else {
          MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                                   << "', each dimension of input shape must be greater than 0, but got input shape "
                                   << i << ": " << input_shape_ptr[i] << ".";
        }
      }
    } else {
      MS_EXCEPTION(TypeError) << "For '" << primitive->name()
                              << "', the dtype of input1 must be in [int32, int64], but got: "
                              << input_type_element->type_id() << ".";
    }
    if (shape_m > max_length) {
      MS_EXCEPTION(ValueError)
        << "For '" << primitive->name()
        << "', the number of elements of output must be less than 'max_length', but got number of elements: " << shape_m
        << ", 'max_length': " << max_length << ".";
    }
    return std::make_shared<abstract::Shape>(out_shape);
  } else {
    const uint32_t input_shapes = static_cast<uint32_t>(std::pow(max_length, 1.0 / shape_v[0]));
    std::vector<int64_t> output_shape;
    ShapeVector shape_min;
    ShapeVector shape_max;
    for (int i = 0; i < shape_v[0]; i++) {
      output_shape.push_back(abstract::Shape::kShapeDimAny);
      shape_min.push_back(0);
      shape_max.push_back(input_shapes);
    }
    return std::make_shared<abstract::Shape>(output_shape, shape_min, shape_max);
  }
}

TypePtr FillV2InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  // Check the data type of the first input
  auto input1 = input_args[kInputIndex0];
  auto input1_type = input1->BuildType();
  MS_EXCEPTION_IF_NULL(input1);
  if (input1->isa<abstract::AbstractTensor>()) {
    const std::set<TypePtr> input1_valid_types = {kInt32, kInt64};
    (void)CheckAndConvertUtils::CheckTensorTypeValid("input1 datatype", input1_type, input1_valid_types, prim_name);
  } else {
    MS_EXCEPTION(TypeError) << "For '" << primitive->name()
                            << "', the dtype of input1 must be in [int32, int64], but got: " << input1_type->ToString()
                            << ".";
  }
  // Check the data type of the second input and infer the data type of the output from the second input
  auto input2 = input_args[kInputIndex1];
  auto input2_type = input2->BuildType();
  MS_EXCEPTION_IF_NULL(input2);
  if (input2->isa<abstract::AbstractTensor>()) {
    auto output_valid_types = common_valid_types;
    (void)output_valid_types.insert(kBool);
    (void)CheckAndConvertUtils::CheckTensorTypeValid("output datatype", input2_type, output_valid_types, prim_name);
  } else {
    MS_EXCEPTION(TypeError)
      << "For '" << prim_name
      << "', the dtype of input2 must be in [bool, int8, int16, int32, int64, uint8, uint16, uint32, "
         "uint64, float16, float32, float64], but got: "
      << input2_type->ToString() << ".";
  }
  auto input2_tensor_type = (input2_type->cast<TensorTypePtr>())->element();

  return input2_tensor_type;
}

AbstractBasePtr FillV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, input_num, prim_name);
  auto infer_type = FillV2InferType(primitive, input_args);
  auto infer_shape = FillV2InferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

REGISTER_PRIMITIVE_EVAL_IMPL(FillV2, prim::kPrimFillV2, FillV2Infer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
