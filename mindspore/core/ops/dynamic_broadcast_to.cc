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

#include "ops/dynamic_broadcast_to.h"

#include <set>
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, 2, prim_name);
  auto input_y = input_args[1];
  MS_EXCEPTION_IF_NULL(input_y);
  abstract::ShapePtr y_shape;
  auto y_value = input_y->BuildValue();
  MS_EXCEPTION_IF_NULL(y_value);
  if (input_y->isa<abstract::AbstractTensor>()) {
    if (y_value->isa<tensor::Tensor>()) {
      auto shape_value = CheckAndConvertUtils::CheckTensorIntValue("shape", y_value, prim_name);
      return std::make_shared<abstract::Shape>(shape_value);
    }
    y_shape = CheckAndConvertUtils::GetTensorInputShape(prim_name, input_args, 1);
    auto shape_value = y_shape->shape();
    if (shape_value.size() != 1) {
      MS_EXCEPTION(TypeError) << "shape size error: " << shape_value.size();
    }
    std::vector<int64_t> output_shape;
    std::vector<int64_t> max_shape;
    std::vector<int64_t> min_shape;
    if (y_shape->IsDynamic()) {
      // max shape unknown
      output_shape.push_back(-2);
    } else {
      auto out_dims = LongToSize(y_shape->shape()[0]);
      for (size_t i = 0; i < out_dims; i++) {
        output_shape.push_back(-1);
      }
      auto min_value = input_y->cast<abstract::AbstractTensorPtr>()->get_min_value();
      auto max_value = input_y->cast<abstract::AbstractTensorPtr>()->get_max_value();
      if (!min_value || !max_value) {
        MS_EXCEPTION(ValueError) << "For BroadcastTo, inputs['shape'] min or max value is empty.";
      }
      min_shape = GetValue<std::vector<int64_t>>(min_value);
      max_shape = GetValue<std::vector<int64_t>>(max_value);
      if (min_shape.size() != out_dims || max_shape.size() != out_dims) {
        MS_EXCEPTION(ValueError) << "For BroadcastTo, inputs['shape'] min or max value not match with out dims.";
      }
    }
    return std::make_shared<abstract::Shape>(output_shape, min_shape, max_shape);
  } else if (input_y->isa<abstract::AbstractTuple>()) {
    auto out_shape = GetValue<std::vector<int64_t>>(y_value);
    return std::make_shared<abstract::Shape>(out_shape);
  }
  MS_EXCEPTION(TypeError) << "For BroadcastTo, input args must be tensor or tuple.";
}

TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto x_dtype = input_args[0]->BuildType()->cast<TensorTypePtr>();
  std::set<TypePtr> template_types = {kTensorType};
  CheckAndConvertUtils::CheckSubClass("x_dtype", x_dtype, template_types, prim->name());
  return x_dtype->element();
}
}  // namespace

AbstractBasePtr DynamicBroadcastToInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) {
  return abstract::MakeAbstract(InferShape(primitive, input_args), InferType(primitive, input_args));
}
REGISTER_PRIMITIVE_EVAL_IMPL(DynamicBroadcastTo, prim::kPrimDynamicBroadcastTo, DynamicBroadcastToInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
