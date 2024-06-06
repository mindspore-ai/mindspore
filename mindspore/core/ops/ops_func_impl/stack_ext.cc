/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "ops/ops_func_impl/stack_ext.h"

#include <vector>
#include <memory>
#include "abstract/abstract_value.h"
#include "ir/dtype/tensor_type.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/shape_utils.h"
#include "utils/log_adapter.h"
#include "ir/primitive.h"
#include "abstract/dshape.h"
#include "base/base.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore::ops {
namespace {
constexpr size_t kTupleInputNum = 2;
inline ShapeVector CheckAndGetInferredShape(const PrimitivePtr &primitive, const ShapeArray &element_shapes) {
  MS_CHECK_VALUE(element_shapes.size() >= 1, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                               "size of elements", element_shapes.size(), kGreaterEqual, 1, primitive));
  bool has_rank_valid_shape = false;
  ShapeVector inferred_shape = {abstract::TensorShape::kShapeRankAny};
  for (auto shape : element_shapes) {
    if (MS_UNLIKELY(IsDynamicRank(shape))) {
      continue;
    }

    if (!has_rank_valid_shape) {
      has_rank_valid_shape = true;
      inferred_shape = shape;
      continue;
    }

    if (MS_UNLIKELY(shape.size() != inferred_shape.size())) {
      MS_EXCEPTION(ValueError) << "All input shape size must be the same!";
    }

    for (size_t j = 0; j < inferred_shape.size(); ++j) {
      if (MS_UNLIKELY(inferred_shape[j] == abstract::TensorShape::kShapeDimAny &&
                      shape[j] != abstract::TensorShape::kShapeDimAny)) {
        inferred_shape[j] = shape[j];
        continue;
      }
      if (shape[j] != abstract::TensorShape::kShapeDimAny && shape[j] != inferred_shape[j]) {
        MS_EXCEPTION(ValueError) << "All input shape must be the same! " << shape << " And " << inferred_shape;
      }
    }
  }

  return inferred_shape;
}
}  // namespace
BaseShapePtr StackExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) const {
  ShapeVector inferred_shape;
  size_t axis_index = input_args.size() - 1;
  auto num = abstract::TensorShape::kShapeDimAny;
  if (MS_LIKELY(input_args.size() == kTupleInputNum)) {
    auto input_shape = input_args[kInputIndex0]->GetShape();
    auto dynamic_sequence = input_shape->cast<abstract::DynamicSequenceShapePtr>();
    if (MS_UNLIKELY(dynamic_sequence != nullptr)) {
      auto key_shape = dynamic_sequence->element_shape();
      inferred_shape = key_shape->GetShapeVector();
    } else {
      auto tuple_shape = input_shape->cast<abstract::TupleShapePtr>();
      MS_EXCEPTION_IF_NULL(tuple_shape);
      num = SizeToLong(tuple_shape->size());
      ShapeArray element_shapes;
      element_shapes.reserve(tuple_shape->size());
      for (size_t i = 0; i < tuple_shape->size(); ++i) {
        element_shapes.push_back((*tuple_shape)[i]->GetShapeVector());
      }
      inferred_shape = CheckAndGetInferredShape(primitive, element_shapes);
    }
  } else if (input_args.size() > kTupleInputNum) {
    // Here for the case expanding tuple to tensors, it maybe deleted in the near future...
    num = SizeToLong(axis_index);
    ShapeArray element_shapes;
    element_shapes.reserve(axis_index);
    for (size_t i = 0; i < axis_index; ++i) {
      element_shapes.push_back(input_args[i]->GetShape()->GetShapeVector());
    }
    inferred_shape = CheckAndGetInferredShape(primitive, element_shapes);
  } else {
    MS_EXCEPTION(ValueError) << "Stack input size(" << input_args.size() << ") is not valid!";
  }

  if (MS_UNLIKELY(IsDynamicRank(inferred_shape))) {
    return std::make_shared<abstract::TensorShape>(inferred_shape);
  }

  size_t out_rank = inferred_shape.size() + 1;

  auto axis_value = input_args[axis_index]->GetValue();
  auto axis_res = GetScalarValue<int64_t>(axis_value);
  if (MS_UNLIKELY(!axis_res.has_value())) {
    ShapeVector res_shape(out_rank, abstract::TensorShape::kShapeDimAny);
    return std::make_shared<abstract::TensorShape>(res_shape);
  }
  auto axis_temp = axis_res.value();
  MS_CHECK_VALUE(-SizeToLong(out_rank) <= axis_temp && axis_temp <= SizeToLong(out_rank) - 1,
                 CheckAndConvertUtils::FormatCheckInRangeMsg(
                   "dim", axis_temp, kIncludeBoth, {-SizeToLong(out_rank), SizeToLong(out_rank) - 1}, primitive));

  auto axis = axis_temp < 0 ? SizeToLong(out_rank) + axis_temp : axis_temp;
  (void)inferred_shape.insert(inferred_shape.begin() + axis, num);
  return std::make_shared<abstract::TensorShape>(inferred_shape);
}

TypePtr StackExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const {
  auto input_type = input_args[kInputIndex0]->GetType();
  if (MS_UNLIKELY(input_args.size() != kTupleInputNum)) {
    return input_type->Clone();
  }

  auto tuple_type = input_type->cast<TuplePtr>();
  MS_EXCEPTION_IF_NULL(tuple_type);
  auto elements = tuple_type->elements();
  MS_CHECK_VALUE(elements.size() >= 1, CheckAndConvertUtils::FormatCheckIntegerMsg("size of elements", elements.size(),
                                                                                   kGreaterEqual, 1, primitive));
  MS_EXCEPTION_IF_NULL(elements[0]);
  auto first_element = elements[0]->cast<TensorTypePtr>();
  if (MS_UNLIKELY(first_element == nullptr)) {
    MS_EXCEPTION(TypeError) << "Infer type failed.";
  }
  auto out_type = first_element->element();
  MS_EXCEPTION_IF_NULL(out_type);

  if (MS_LIKELY(!tuple_type->dynamic_len())) {
    for (size_t i = 1; i < elements.size(); ++i) {
      MS_EXCEPTION_IF_NULL(elements[i]);
      auto cur_element = elements[i]->cast<TensorTypePtr>();
      MS_EXCEPTION_IF_NULL(cur_element);
      auto element_type = cur_element->element();
      MS_EXCEPTION_IF_NULL(element_type);
      if (out_type->ToString() != element_type->ToString()) {
        MS_EXCEPTION(TypeError) << "All input must have the same data type(input[" << i
                                << "] data type = " << element_type->ToString()
                                << ", the first type= " << out_type->ToString() << ")!";
      }
    }
  }
  return std::make_shared<TensorType>(out_type);
}

ShapeArray StackExtFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  auto tuple_x = input_values[kInputIndex0]->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(tuple_x);
  ShapeArray shapes;
  shapes.reserve(tuple_x->size());
  for (const auto &item : tuple_x->value()) {
    MS_EXCEPTION_IF_NULL(item);
    auto tensor = item->cast<tensor::BaseTensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    shapes.push_back(tensor->shape());
  }
  auto num = SizeToLong(tuple_x->size());
  auto inferred_shape = CheckAndGetInferredShape(primitive, shapes);
  auto axis = GetValue<int64_t>(input_values[kInputIndex1]);
  size_t out_rank = inferred_shape.size() + 1;
  MS_CHECK_VALUE(-SizeToLong(out_rank) <= axis && axis <= SizeToLong(out_rank) - 1,
                 CheckAndConvertUtils::FormatCheckInRangeMsg(
                   "dim", axis, kIncludeBoth, {-SizeToLong(out_rank), SizeToLong(out_rank) - 1}, primitive));

  axis = axis < 0 ? SizeToLong(out_rank) + axis : axis;
  (void)inferred_shape.insert(inferred_shape.begin() + axis, num);
  return {inferred_shape};
}

TypePtrList StackExtFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  auto tuple_x = input_values[kInputIndex0]->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(tuple_x);
  MS_CHECK_VALUE(tuple_x->size() >= 1, CheckAndConvertUtils::FormatCheckIntegerMsg("size of elements", tuple_x->size(),
                                                                                   kGreaterEqual, 1, primitive));
  TypePtrList elements;
  elements.reserve(tuple_x->size());
  for (const auto &item : tuple_x->value()) {
    MS_EXCEPTION_IF_NULL(item);
    auto tensor = item->cast<tensor::BaseTensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    elements.push_back(tensor->Dtype());
  }

  auto out_type = elements[0];
  MS_EXCEPTION_IF_NULL(out_type);

  // Check all element' types is valid:
  // 1. all same
  // 2. is one of the common_valid_types_with_complex_and_bool.
  if (MS_UNLIKELY(std::any_of(elements.cbegin(), elements.cend(),
                              [&out_type](const TypePtr &type) { return type != out_type; }))) {
    std::ostringstream buffer;
    buffer << "The primitive[" << primitive->name() << "]'s input arguments must be same, but got ";
    for (size_t i = 0; i < elements.size(); ++i) {
      MS_EXCEPTION_IF_NULL(elements[i]);
      buffer << "element[" << i << "]:" << elements[i]->ToString();
      if (i != (elements.size() - 1)) {
        buffer << ", ";
      }
    }
    buffer << ".";
    MS_LOG(EXCEPTION) << buffer.str();
  }
  if (MS_UNLIKELY(std::all_of(common_valid_types_with_complex_and_bool.cbegin(),
                              common_valid_types_with_complex_and_bool.cend(),
                              [&out_type](const TypePtr &type) { return out_type != type; }))) {
    std::ostringstream buffer;
    buffer << "The primitive[" << primitive->name() << "]'s valid type list: {";
    for (const auto &type : common_valid_types_with_complex_and_bool) {
      buffer << type->ToString();
      if (type != *(--common_valid_types_with_complex_and_bool.end())) {
        buffer << ", ";
      }
    }
    buffer << "}, but got " << out_type->ToString();
    MS_LOG(EXCEPTION) << buffer.str();
  }

  return {out_type};
}

REGISTER_SIMPLE_INFER(kNameStackExt, StackExtFuncImpl)
}  // namespace mindspore::ops
