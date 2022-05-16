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
#include "ops/unsorted_segment_arithmetic.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"
#include "ops/unsorted_segment_max.h"
#include "ops/unsorted_segment_min.h"
#include "ops/unsorted_segment_prod.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr UnsortedSegmentArithmeticInferShape(const PrimitivePtr &primitive,
                                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto x_shape_ptr = input_args[kInputIndex0]->BuildShape();
  MS_EXCEPTION_IF_NULL(x_shape_ptr);
  auto segment_ids_shape_ptr = input_args[kInputIndex1]->BuildShape();
  MS_EXCEPTION_IF_NULL(segment_ids_shape_ptr);
  auto num_segments_shape_ptr = input_args[kInputIndex2]->BuildShape();
  MS_EXCEPTION_IF_NULL(num_segments_shape_ptr);

  auto num_segments = input_args[kInputIndex2]->cast<abstract::AbstractScalarPtr>();
  int64_t num_segments_value = 0;

  if (num_segments != nullptr && num_segments->BuildValue() != kAnyValue) {
    num_segments_value = GetValue<int64_t>(num_segments->BuildValue());
    primitive->AddAttr(kNumSegments, MakeValue(num_segments_value));
  }

  if (x_shape_ptr->IsDynamic() || segment_ids_shape_ptr->IsDynamic() || num_segments_shape_ptr->IsDynamic()) {
    return x_shape_ptr->cast<abstract::ShapePtr>();
  }

  if (num_segments == nullptr || num_segments->BuildValue() == kAnyValue) {
    auto value_ptr = primitive->GetAttr(kNumSegments);
    if (value_ptr != nullptr) {
      num_segments_value = GetValue<int64_t>(value_ptr);
    } else {
      return x_shape_ptr->cast<abstract::ShapePtr>();
    }
  }

  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto ids_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  if (x_shape.size() < ids_shape.size()) {
    MS_LOG(ERROR) << "For " << prim_name << ", invalid input_args and segment_ids shape size";
    return input_args[kInputIndex0]->BuildShape()->cast<abstract::ShapePtr>();
  }

  for (size_t i = 0; i < ids_shape.size(); i++) {
    if (x_shape[i] != ids_shape[i]) {
      MS_LOG(ERROR) << "For " << prim_name << ", invalid input_args and segment_ids shape[" << i << "]: " << x_shape[i]
                    << ", " << ids_shape[i];
      return x_shape_ptr->cast<abstract::ShapePtr>();
    }
  }

  std::vector<int64_t> out_shape;
  out_shape.push_back(num_segments_value);
  for (size_t i = ids_shape.size(); i < x_shape.size(); i++) {
    out_shape.push_back(x_shape.at(i));
  }
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr UnsortedSegmentArithmeticInferType(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto in_type_ptr = input_args[kInputIndex0]->BuildType();
  MS_EXCEPTION_IF_NULL(in_type_ptr);
  std::set<TypePtr> in_type_set = {kFloat16, kFloat32, kInt32};
  return CheckAndConvertUtils::CheckTensorTypeValid("x", in_type_ptr, in_type_set, prim_name);
}
}  // namespace

AbstractBasePtr UnsortedSegmentArithmeticInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputNum = 3;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, primitive->name());
  auto infer_type = UnsortedSegmentArithmeticInferType(primitive, input_args);
  auto infer_shape = UnsortedSegmentArithmeticInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(UnsortedSegmentMax, BaseOperator);
MIND_API_OPERATOR_IMPL(UnsortedSegmentMin, BaseOperator);
MIND_API_OPERATOR_IMPL(UnsortedSegmentProd, BaseOperator);

REGISTER_PRIMITIVE_EVAL_IMPL(UnsortedSegmentMax, prim::kPrimUnsortedSegmentMax, UnsortedSegmentArithmeticInfer, nullptr,
                             true);
REGISTER_PRIMITIVE_EVAL_IMPL(UnsortedSegmentMin, prim::kPrimUnsortedSegmentMin, UnsortedSegmentArithmeticInfer, nullptr,
                             true);
REGISTER_PRIMITIVE_EVAL_IMPL(UnsortedSegmentProd, prim::kPrimUnsortedSegmentProd, UnsortedSegmentArithmeticInfer,
                             nullptr, true);
}  // namespace ops
}  // namespace mindspore
