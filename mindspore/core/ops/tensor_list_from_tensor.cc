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

#include "ops/tensor_list_from_tensor.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr TensorListFromTensorInferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto tensor_list_from_tensor_prim = primitive->cast<PrimTensorListFromTensorPtr>();
  MS_EXCEPTION_IF_NULL(tensor_list_from_tensor_prim);
  auto prim_name = tensor_list_from_tensor_prim->name();
  auto input0_shape =
    CheckAndConvertUtils::ConvertShapePtrToShape("input0 shape", input_args[0]->BuildShape(), prim_name);
  auto input1_shape =
    CheckAndConvertUtils::ConvertShapePtrToShape("input1 shape", input_args[1]->BuildShape(), prim_name);
  if (input0_shape.size() < 1) {
    MS_LOG(ERROR) << "input0_shape.size():" << input0_shape.size() << " must be greater than 0!";
  }
  int64_t dim0 = input0_shape[0];
  if (dim0 < 0) {
    MS_LOG(ERROR) << "input[0] dim0:" << dim0 << " must be greater than or equal to 0!";
  }
  auto input1 = &input1_shape[0];
  MS_ASSERT(input1 != nullptr);
  if (input1 == nullptr) {
    MS_LOG(ERROR) << "input1 is nullptr";
  }
  std::vector<int64_t> infer_shape = {1, dim0};
  return std::make_shared<abstract::Shape>(infer_shape);
}

TypePtr TensorListFromTensorInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  TypeId infer_type = kObjectTypeTensorType;
  return TypeIdToType(infer_type);
}
}  // namespace

void TensorListFromTensor::Init(const int64_t element_dtype, const int64_t shape_type) {
  this->set_element_dtype(element_dtype);
  this->set_shape_type(shape_type);
}

int64_t TensorListFromTensor::get_element_dtype() const {
  auto value_ptr = GetAttr(kElement_dtype);
  return GetValue<int64_t>(value_ptr);
}

int64_t TensorListFromTensor::get_shape_type() const {
  auto value_ptr = GetAttr(kShapeType);
  return GetValue<int64_t>(value_ptr);
}

void TensorListFromTensor::set_element_dtype(const int64_t element_dtype) {
  this->AddAttr(kElement_dtype, MakeValue(element_dtype));
}

void TensorListFromTensor::set_shape_type(const int64_t shape_type) {
  this->AddAttr(kShapeType, MakeValue(shape_type));
}

AbstractBasePtr TensorListFromTensorInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(TensorListFromTensorInferType(primitive, input_args),
                                                    TensorListFromTensorInferShape(primitive, input_args)->shape());
}
REGISTER_PRIMITIVE_C(kNameTensorListFromTensor, TensorListFromTensor);
}  // namespace ops
}  // namespace mindspore
