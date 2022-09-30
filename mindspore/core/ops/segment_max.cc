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

#include "ops/segment_max.h"
#include <algorithm>
#include <set>
#include "ops/op_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr SegmentMaxInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto max_length_ptr = primitive->GetAttr("max_length");
  MS_EXCEPTION_IF_NULL(max_length_ptr);
  int64_t max_length = GetValue<int64_t>(max_length_ptr);
  auto x_shape_ptr = input_args[0]->BuildShape();
  MS_EXCEPTION_IF_NULL(x_shape_ptr);
  auto segment_ids_shape_ptr = input_args[1]->BuildShape();
  MS_EXCEPTION_IF_NULL(segment_ids_shape_ptr);
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x_shape_ptr)[kShape];
  if (x_shape.size() == 0) {
    MS_EXCEPTION(ValueError) << "The rank of input_x must not be less than 1, but got 0.";
  }
  auto segment_ids_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(segment_ids_shape_ptr)[kShape];
  ShapeVector out_shape(x_shape);
  auto segment_ids_ptr = input_args[1]->BuildValue();
  MS_EXCEPTION_IF_NULL(segment_ids_ptr);
  if (!segment_ids_ptr->isa<AnyValue>() && !segment_ids_ptr->isa<None>()) {
    if (segment_ids_shape.size() != 1) {
      MS_EXCEPTION(ValueError) << "Segment_ids must be a 1D tensor, but got " << segment_ids_shape.size() << "D tensor";
    }
    if (segment_ids_shape[0] != x_shape[0]) {
      MS_EXCEPTION(ValueError)
        << "The amount of data for segment_ids must be equal to the first dimension of the shape of input_x, but got "
        << segment_ids_shape[0] << " and " << x_shape[0] << ".";
    }
    auto segment_ids_tensor = segment_ids_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(segment_ids_tensor);
    auto data_size = segment_ids_tensor->DataSize();
    auto segment_ids_type_id = segment_ids_tensor->data_type();
    if (segment_ids_type_id == kNumberTypeInt64) {
      int64_t *segment_ids_data = reinterpret_cast<int64_t *>(segment_ids_tensor->data_c());
      if (segment_ids_data[0] < 0) {
        MS_EXCEPTION(ValueError) << "The values of segment_ids must be nonnegative. but got " << segment_ids_data[0]
                                 << ".";
      }
      for (size_t i = 0; i < data_size - 1; ++i) {
        if (segment_ids_data[i] > segment_ids_data[i + 1]) {
          MS_EXCEPTION(ValueError) << "segment_ids must be a tensor with element values sorted in ascending order.";
        }
      }
      out_shape[0] = segment_ids_data[data_size - 1] + 1;
    } else if (segment_ids_type_id == kNumberTypeInt32) {
      int32_t *segment_ids_data = reinterpret_cast<int32_t *>(segment_ids_tensor->data_c());
      if (segment_ids_data[0] < 0) {
        MS_EXCEPTION(ValueError) << "The values of segment_ids must be nonnegative. but got " << segment_ids_data[0]
                                 << ".";
      }
      for (size_t i = 0; i < data_size - 1; ++i) {
        if (segment_ids_data[i] > segment_ids_data[i + 1]) {
          MS_EXCEPTION(ValueError) << "segment_ids must be a tensor with element values sorted in ascending order.";
        }
      }
      out_shape[0] = IntToLong(segment_ids_data[data_size - 1] + 1);
    }
    int64_t length = 1;
    for (size_t i = 0; i < out_shape.size(); ++i) {
      length *= static_cast<int64_t>(out_shape[i]);
    }
    if (length > max_length) {
      MS_EXCEPTION(ValueError) << "The number of elements of output must be less than max length: " << max_length
                               << ", but got " << length
                               << "! The shape of output should be reduced or max_length should be increased";
    }
    return std::make_shared<abstract::Shape>(out_shape);
  } else {
    int64_t length = 1;
    for (size_t i = 1; i < x_shape.size(); ++i) {
      length *= static_cast<int64_t>(x_shape[i]);
    }
    const uint32_t max_shape_value = LongToUint(max_length / length);
    ShapeVector min_shape(x_shape);
    ShapeVector max_shape(x_shape);
    out_shape[0] = abstract::Shape::kShapeDimAny;
    min_shape[0] = 1;
    max_shape[0] = max_shape_value;
    return std::make_shared<abstract::Shape>(out_shape, min_shape, max_shape);
  }
}

TypePtr SegmentMaxInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  TypePtr x_type = input_args[0]->BuildType();
  TypePtr segment_ids_type = input_args[1]->BuildType();
  const std::set<TypePtr> x_valid_types = {kFloat16, kFloat32, kFloat64, kInt8,   kInt16, kInt32,
                                           kInt64,   kUInt8,   kUInt16,  kUInt32, kUInt64};
  const std::set<TypePtr> segment_ids_valid_types = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x_type", x_type, x_valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("segment_ids_type", segment_ids_type, segment_ids_valid_types,
                                                   prim_name);
  return x_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(SegmentMax, BaseOperator);
AbstractBasePtr SegmentMaxInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto type = SegmentMaxInferType(primitive, input_args);
  auto shape = SegmentMaxInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(SegmentMax, prim::kPrimSegmentMax, SegmentMaxInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
