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

#include <set>
#include <vector>
#include <memory>
#include <map>
#include <string>
#include <climits>
#include "ops/sparse_slice.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
void SparseSliceCheckInputTensor(const std::vector<AbstractBasePtr> &input_args) {
  auto indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto values_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto shape_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto start_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
  auto size_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex4]->BuildShape())[kShape];
  if (indices_shape.size() != kDim2) {
    MS_EXCEPTION(ValueError) << "For SparseSlice, indices should be a 2-D tensor"
                             << ", while input_indices dim num is " << indices_shape.size() << ".";
  }
  if (indices_shape[1] != kDim2) {
    MS_EXCEPTION(ValueError) << "For SparseSlice, indices shape should be (2, n)"
                             << ", while input_indices shape dim0 is " << indices_shape[0] << ".";
  }
  if (values_shape.size() != kDim1) {
    MS_EXCEPTION(ValueError) << "For SparseSlice, values should be a 1-D tensor"
                             << ",  while input_values dim num is " << values_shape.size() << ".";
  }
  if (indices_shape[0] != values_shape[0]) {
    MS_EXCEPTION(ValueError) << "For SparseSlice"
                             << ", dim1 size of `indices` and dim0 size of `values` should be the same"
                             << " while indices_shape dim1 size is " << indices_shape[1]
                             << ", values_shape dim0 size is " << values_shape[0] << ".";
  }
  if (shape_shape.size() != kDim1) {
    MS_EXCEPTION(ValueError) << "For SparseSlice"
                             << ", shape should be a 1-D tensor, while input_shape dim num is " << shape_shape.size()
                             << ".";
  }
  if (shape_shape[0] != kDim2) {
    MS_EXCEPTION(ValueError) << "For SparseSlice"
                             << ", the shape of input shape should be [2] but got shape [" << shape_shape[0] << "].";
  }
  if (start_shape[0] != kDim2) {
    MS_EXCEPTION(ValueError) << "For SparseSlice, start should be a 2-D tensor"
                             << ", while dim num is " << start_shape.size() << ".";
  }
  if (size_shape[0] != kDim2) {
    MS_EXCEPTION(ValueError) << "For SparseSlice, size should be a 2-D tensor"
                             << ", while dim num is " << size_shape.size() << ".";
  }
}

template <typename T>
void SparseSliceIndicesBoundCheck(T *indices_val, size_t indices_num, T *shape_val, std::string name) {
  if (shape_val[0] <= 0 || shape_val[1] <= 0) {
    MS_EXCEPTION(ValueError) << "For SparseSlice, " << name << "_shape should be positive, "
                             << "while got shape [" << shape_val[0] << ", " << shape_val[1] << "].";
  }
  size_t half_num = indices_num / kDim2;
  for (size_t i = 0; i < half_num; i++) {
    if ((indices_val[i] < 0) || (indices_val[i] >= shape_val[0])) {
      MS_EXCEPTION(ValueError) << "For SparseSlice, " << name << "_indices row index should between [0, "
                               << shape_val[0] << "], while got row index " << indices_val[i] << ".";
    }
    if ((indices_val[i + half_num] < 0) || (indices_val[i + half_num] >= shape_val[1])) {
      MS_EXCEPTION(ValueError) << "For SparseSlice, " << name << "_indices col index should between [0, "
                               << shape_val[1] << "], while got col index " << indices_val[i + half_num] << ".";
    }
  }
}

void SparseSliceCheckIndices(const std::vector<AbstractBasePtr> &input_args) {
  auto x1_indices_abstract = input_args[kInputIndex0]->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(x1_indices_abstract);
  auto x1_indices_value_ptr = x1_indices_abstract->BuildValue();
  MS_EXCEPTION_IF_NULL(x1_indices_value_ptr);
  auto x1_indices_tensor = x1_indices_value_ptr->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x1_indices_tensor);
  auto x1_indices_type = input_args[kInputIndex0]->BuildType();
  MS_EXCEPTION_IF_NULL(x1_indices_type);
  auto x1_indices_type_id = x1_indices_type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(x1_indices_type_id);
  auto x1_indices_type_element = x1_indices_type_id->element();
  MS_EXCEPTION_IF_NULL(x1_indices_type_element);
  auto x1_shape_abstract = input_args[kInputIndex2]->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(x1_shape_abstract);
  auto x1_shape_value_ptr = x1_shape_abstract->BuildValue();
  MS_EXCEPTION_IF_NULL(x1_shape_value_ptr);
  auto x1_shape_tensor = x1_shape_value_ptr->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x1_shape_tensor);
  if (x1_indices_type_element->type_id() == kNumberTypeInt32) {
    SparseSliceIndicesBoundCheck<int32_t>(reinterpret_cast<int32_t *>(x1_indices_tensor->data_c()),
                                          x1_indices_tensor->DataSize(),
                                          reinterpret_cast<int32_t *>(x1_shape_tensor->data_c()), "x1");
  } else {
    SparseSliceIndicesBoundCheck<int64_t>(reinterpret_cast<int64_t *>(x1_indices_tensor->data_c()),
                                          x1_indices_tensor->DataSize(),
                                          reinterpret_cast<int64_t *>(x1_shape_tensor->data_c()), "x1");
  }
}

abstract::TupleShapePtr SparseSliceInferShape(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  if (input_args[kInputIndex0]->isa<abstract::AbstractTensor>() &&
      !input_args[kInputIndex0]->BuildValue()->isa<AnyValue>() &&
      !input_args[kInputIndex0]->BuildValue()->isa<None>()) {
    SparseSliceCheckIndices(input_args);
    auto input_indices_abstract = input_args[kInputIndex0]->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(input_indices_abstract);
    auto input_indices_value_ptr = input_indices_abstract->BuildValue();
    MS_EXCEPTION_IF_NULL(input_indices_value_ptr);
    auto input_indices_tensor = input_indices_value_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(input_indices_tensor);
    auto input_indices_val = reinterpret_cast<int64_t *>(input_indices_tensor->data_c());
    auto input_start_ptr = input_args[kInputIndex3]->BuildValue();
    MS_EXCEPTION_IF_NULL(input_start_ptr);
    auto input_start_tensor = input_start_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(input_start_tensor);
    auto input_start_val = reinterpret_cast<int64_t *>(input_start_tensor->data_c());
    auto input_size_ptr = input_args[kInputIndex4]->BuildValue();
    MS_EXCEPTION_IF_NULL(input_size_ptr);
    auto input_size_tensor = input_size_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(input_size_tensor);
    auto input_size_val = reinterpret_cast<int64_t *>(input_size_tensor->data_c());
    int64_t count = 0;
    int64_t size_left = input_size_val[0];
    int64_t size_right = input_size_val[1];
    int64_t low = input_start_val[0] + size_left;
    int64_t high = input_start_val[1] + size_right;
    for (size_t i = 0; i < input_indices_tensor->DataSize(); i = i + kInputIndex2) {
      if ((input_indices_val[i] >= input_start_val[0] && input_indices_val[i] < low) &&
          (input_indices_val[i + 1] >= input_start_val[1] && input_indices_val[i + 1] < high)) {
        count = count + 1;
      }
    }
    std::vector<int64_t> output_indices_shape = {count, kInputIndex2};
    abstract::ShapePtr output_indices_shape_list =
      std::make_shared<abstract::Shape>(output_indices_shape, output_indices_shape, output_indices_shape);
    std::vector<int64_t> output_values_shape = {count};
    abstract::ShapePtr output_values_shape_list =
      std::make_shared<abstract::Shape>(output_values_shape, output_values_shape, output_values_shape);
    // std::vector<int64_t> output_size_shape = {size_left, size_right};
    std::vector<int64_t> output_size_shape = {kInputIndex2};
    abstract::ShapePtr output_size_shape_list =
      std::make_shared<abstract::Shape>(output_size_shape, output_size_shape, output_size_shape);
    return std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{output_indices_shape_list, output_values_shape_list, output_size_shape_list});
  } else {
    std::vector<int64_t> output_indices_shape = {abstract::Shape::kShapeDimAny, 2};
    abstract::ShapePtr output_indices_shape_list =
      std::make_shared<abstract::Shape>(output_indices_shape, output_indices_shape, output_indices_shape);
    std::vector<int64_t> output_values_shape = {abstract::Shape::kShapeDimAny};
    abstract::ShapePtr output_values_shape_list =
      std::make_shared<abstract::Shape>(output_values_shape, output_values_shape, output_values_shape);
    std::vector<int64_t> output_size_shape = {abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny};
    abstract::ShapePtr output_size_shape_list =
      std::make_shared<abstract::Shape>(output_size_shape, output_size_shape, output_size_shape);
    return std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{output_indices_shape_list, output_values_shape_list, output_size_shape_list});
  }
}

TuplePtr SparseSliceInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = prim->name();
  std::map<std::string, TypePtr> types;
  (void)types.emplace("indices", input_args[kInputIndex0]->BuildType());
  (void)types.emplace("shape", input_args[kInputIndex2]->BuildType());
  (void)types.emplace("start", input_args[kInputIndex3]->BuildType());
  (void)types.emplace("size", input_args[kInputIndex4]->BuildType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, {kInt64, kInt32}, prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("values", input_args[kInputIndex1]->BuildType(),
                                                   {kUInt8, kInt8, kInt16, kInt32, kInt64, kFloat32, kFloat64},
                                                   op_name);
  std::map<std::string, TypePtr> args = {{"values", input_args[kInputIndex1]->BuildType()}};
  auto output_values_type = CheckAndConvertUtils::CheckTensorTypeSame(
    args, {kInt8, kInt16, kInt32, kInt64, kUInt8, kFloat32, kFloat64}, op_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{kInt64, output_values_type, kInt64});
}
}  // namespace

AbstractBasePtr SparseSliceInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 5;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  SparseSliceCheckInputTensor(input_args);
  auto infer_type = SparseSliceInferType(primitive, input_args);
  auto infer_shape = SparseSliceInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

REGISTER_PRIMITIVE_EVAL_IMPL(SparseSlice, prim::kPrimSparseSlice, SparseSliceInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
