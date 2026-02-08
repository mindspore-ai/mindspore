/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/trace_ext.h"

#include <vector>
#include <memory>
#include <set>
#include "primitive/op_name.h"
#include "utils/check_convert_utils.h"
#include "utils/shape_utils.h"
#include "utils/log_adapter.h"
#include "ir/primitive.h"
#include "abstract/dshape.h"
#include "base/base.h"
#include "ops_utils/op_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore::ops {
static inline bool IsValidTraceExtType(TypeId t) {
  static const std::set<TypeId> valid_types = {kNumberTypeBool,     kNumberTypeInt8,      kNumberTypeInt16,
                                               kNumberTypeInt32,    kNumberTypeInt64,     kNumberTypeUInt8,
                                               kNumberTypeFloat16,  kNumberTypeFloat32,   kNumberTypeFloat64,
                                               kNumberTypeBFloat16, kNumberTypeComplex64, kNumberTypeComplex128};
  return valid_types.find(t) != valid_types.end();
}

static inline bool IsIntegralTraceExtType(TypeId t) {
  static const std::set<TypeId> integral_types = {kNumberTypeBool,   kNumberTypeInt8,   kNumberTypeInt16,
                                                  kNumberTypeInt32,  kNumberTypeInt64,  kNumberTypeUInt8,
                                                  kNumberTypeUInt16, kNumberTypeUInt32, kNumberTypeUInt64};
  return integral_types.find(t) != integral_types.end();
}

BaseShapePtr TraceExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  auto base_shape = input_args[kInputIndex0]->GetShape();
  MS_EXCEPTION_IF_NULL(base_shape);
  const auto &shape = base_shape->GetShapeVector();

  const size_t kTraceInputRank = 2;
  if (!IsDynamic(shape) && shape.size() != kTraceInputRank) {
    MS_LOG(EXCEPTION) << "For Primitive[TraceExt], the rank of the input must be 2, but got " << shape.size() << "!";
  }

  return std::make_shared<abstract::Shape>(ShapeVector{});
}

TypePtr TraceExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  auto input_type = input_args[kInputIndex0]->GetType();
  auto element = input_type->cast<TensorTypePtr>()->element();
  MS_EXCEPTION_IF_NULL(element);
  auto input_type_id = element->type_id();
  if (!IsValidTraceExtType(input_type_id)) {
    MS_EXCEPTION(TypeError) << "For Primitive[TraceExt], the type of the input must be [Bool , Uint8, Int8, Int16, "
                               "Int32, Int64, Float16, Float32, Float64, BFloat16, Complex64, Complex128], but got "
                            << input_type << "!";
  }
  if (IsIntegralTraceExtType(input_type_id)) {
    return std::make_shared<TensorType>(kInt64);
  }
  return input_type;
}

ShapeArray TraceExtFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &input = input_values[kIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(input);
  const size_t kTraceInputRank = 2;
  if (input->shape().size() != kTraceInputRank) {
    MS_LOG(EXCEPTION) << "For Primitive[TraceExt], the rank of the input must be 2, but got " << input->shape().size()
                      << "!";
  }
  return {ShapeVector{}};
}

TypePtrList TraceExtFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &input = input_values[kIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(input);
  const auto &input_type = input->Dtype();
  const auto &input_type_id = input->Dtype()->type_id();
  if (!IsValidTraceExtType(input_type_id)) {
    MS_EXCEPTION(TypeError) << "For Primitive[TraceExt], the type of the input must be [Bool , Uint8, Int8, Int16, "
                               "Int32, Int64, Float16, Float32, Float64, BFloat16, Complex64, Complex128], but got "
                            << input_type << "!";
  }
  if (IsIntegralTraceExtType(input_type_id)) {
    return {kInt64};
  }
  return {input_type};
}

REGISTER_SIMPLE_INFER(kNameTraceExt, TraceExtFuncImpl)
}  // namespace mindspore::ops
