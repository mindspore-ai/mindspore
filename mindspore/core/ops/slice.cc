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

#include "ops/slice.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kDynamicOutValue = -2;
std::vector<int64_t> InferImplSliceFuncCalInputValue(const PrimitivePtr &primitive, const ValuePtr &input_value) {
  std::vector<int64_t> tmp_input;
  MS_EXCEPTION_IF_NULL(input_value);
  if (input_value->isa<tensor::Tensor>()) {
    tmp_input = CheckAndConvertUtils::CheckTensorIntValue("slice args value", input_value, primitive->name());
  } else if (input_value->isa<ValueTuple>()) {
    tmp_input = CheckAndConvertUtils::CheckTupleInt("slice args value", input_value, primitive->name());
  } else if (input_value->isa<ValueList>()) {
    tmp_input = CheckAndConvertUtils::CheckListInt("slice args value", input_value, primitive->name());
  } else {
    MS_EXCEPTION(TypeError) << "For Slice, the begin and size must be Tuple or List.";
  }

  return tmp_input;
}

abstract::ShapePtr SliceInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto input_x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto input_size_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape());
  auto input_x_shape = input_x_shape_map[kShape];
  auto input_x_shape_min = input_x_shape_map[kMinShape];
  auto input_x_shape_max = input_x_shape_map[kMaxShape];
  auto input_begin_value_ptr = input_args[kInputIndex1]->BuildValue();
  auto input_size_value_ptr = input_args[kInputIndex2]->BuildValue();
  auto input_size_shape = input_size_shape_map[kShape];
  (void)CheckAndConvertUtils::CheckInteger("rank of input_x", SizeToLong(input_x_shape.size()), kGreaterThan, 0,
                                           prim_name);
  ShapeVector out_shape = {};
  ShapeVector out_shape_min;
  ShapeVector out_shape_max;
  if (input_x_shape[0] == 0) {
    MS_EXCEPTION(ValueError) << "For Slice, the input_x must hava value.";
  }
  if (!input_x_shape_max.empty()) {
    out_shape_min = input_x_shape_min;
    out_shape_max = input_x_shape_max;
  } else {
    out_shape_min = input_x_shape;
    out_shape_max = input_x_shape;
  }
  if (input_begin_value_ptr->isa<AnyValue>() && !input_size_value_ptr->isa<AnyValue>()) {
    auto input_value = input_args[kInputIndex2]->BuildValue();
    auto tmp_input = InferImplSliceFuncCalInputValue(primitive, input_value);
    for (size_t i = 0; i < tmp_input.size(); i++) {
      out_shape.push_back(-1);
    }
    return std::make_shared<abstract::Shape>(out_shape, out_shape_min, out_shape_max);
  }
  if (input_size_value_ptr->isa<AnyValue>()) {
    if (input_size_shape.size() == 0) {
      out_shape.push_back(kDynamicOutValue);
      return std::make_shared<abstract::Shape>(out_shape, out_shape_min, out_shape_max);
    }
    if (input_size_shape[0] < 0) {
      MS_EXCEPTION(ValueError) << "For Slice, the size shape haven't support dynamic yet.";
    }
    for (int64_t i = 0; i < input_size_shape[0]; i++) {
      out_shape.push_back(-1);
    }
    return std::make_shared<abstract::Shape>(out_shape, out_shape_min, out_shape_max);
  }

  auto input_begin_value = InferImplSliceFuncCalInputValue(primitive, input_args[kInputIndex1]->BuildValue());
  auto input_size_value = InferImplSliceFuncCalInputValue(primitive, input_args[kInputIndex2]->BuildValue());
  auto rank = input_x_shape.size();
  if (input_begin_value.size() != rank || input_size_value.size() != rank) {
    MS_EXCEPTION(ValueError) << "For Slice, the shape of input|begin|size must be equal.";
  }
  (void)CheckAndConvertUtils::CheckPositiveVector("input_begin", input_begin_value, prim_name);
  for (size_t i = 0; i < rank; ++i) {
    if (input_x_shape[i] < 0) {
      continue;
    }
    if (input_begin_value[i] + input_size_value[i] > input_x_shape[i]) {
      MS_EXCEPTION(ValueError) << "For Slice, the sum of begin_shape[" << i << "] and size_shape[" << i
                               << "] must be no greater than input_x_shape[" << i << "].";
    }
    if (input_size_value[i] == -1) {
      input_size_value[i] = input_x_shape[i] - input_begin_value[i];
    }
  }
  return std::make_shared<abstract::Shape>(input_size_value);
}

TypePtr SliceInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  return CheckAndConvertUtils::CheckSubClass("input_x", input_args[0]->BuildType(), {kTensorType}, primitive->name());
}
}  // namespace

MIND_API_OPERATOR_IMPL(Slice, BaseOperator);
AbstractBasePtr SliceInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto prim_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputIndex3, prim_name);
  auto type = SliceInferType(primitive, input_args);
  auto shape = SliceInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}

std::vector<int64_t> Slice::get_begin() const {
  auto value_ptr = GetAttr(kBegin);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

std::vector<int64_t> Slice::get_size() const {
  auto value_ptr = GetAttr(kSize);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

REGISTER_HOST_DEPENDS(kNameSlice, {1, 2});
REGISTER_PRIMITIVE_EVAL_IMPL(Slice, prim::kPrimSlice, SliceInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
