/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "ops/unsorted_segment_sum.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <map>
#include <vector>
#include <utility>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "ops/unsorted_segment_arithmetic.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr UnsortedSegmentSumInferShape(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string &op_name = primitive->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto x_shape_rank = SizeToLong(x_shape.size());
  (void)CheckAndConvertUtils::CheckInteger("input_x size", x_shape_rank, kGreaterThan, 0, op_name);
  auto segment_ids_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto segment_ids_shape_rank = SizeToLong(segment_ids_shape.size());
  (void)CheckAndConvertUtils::CheckInteger("segment_ids size", segment_ids_shape_rank, kGreaterThan, 0, op_name);
  ShapeVector output_shape;
  if (IsDynamicRank(x_shape) || IsDynamicRank(segment_ids_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
  }
  (void)CheckAndConvertUtils::CheckValue<size_t>("x rank", x_shape.size(), kGreaterEqual, "segment_ids_shape rank",
                                                 segment_ids_shape.size(), op_name);
  bool x_any_shape =
    std::any_of(x_shape.begin(), x_shape.end(), [](int64_t dim) { return dim == abstract::Shape::kShapeDimAny; });
  bool ids_any_shape = std::any_of(segment_ids_shape.begin(), segment_ids_shape.end(),
                                   [](int64_t dim) { return dim == abstract::Shape::kShapeDimAny; });
  if (!x_any_shape && !ids_any_shape) {
    for (uint64_t i = 0; i < segment_ids_shape.size(); i++) {
      if (segment_ids_shape[i] != x_shape[i]) {
        MS_EXCEPTION(ValueError) << "For '" << op_name
                                 << "', the whose shape of 'segment_ids' must equal to the shape of 'input_x', ids["
                                 << i << "] should be = input[" << i << "]: " << x_shape[i] << ", but got "
                                 << segment_ids_shape[i];
      }
    }
  }
  abstract::CheckShapeAnyAndPositive(op_name + " x_shape", x_shape);
  abstract::CheckShapeAnyAndPositive(op_name + " segment_ids_shape", segment_ids_shape);

  ShapeVector num_vec;
  num_vec.push_back(GetNumSegmentsValue(primitive, input_args));
  int64_t batch_rank = 0;
  if (primitive->HasAttr(kBatchRank)) {
    auto batch_rank_ptr = primitive->GetAttr(kBatchRank);
    batch_rank = GetValue<int64_t>(batch_rank_ptr);
  }
  if (batch_rank != 0) {
    (void)copy(x_shape.begin(), x_shape.begin() + batch_rank, std::back_inserter(output_shape));
  }
  auto calc_shape = [segment_ids_shape_rank](const ShapeVector &num_vec, const ShapeVector &x_shape) -> ShapeVector {
    ShapeVector out_vec;
    (void)copy(num_vec.begin(), num_vec.end(), std::back_inserter(out_vec));
    (void)copy(x_shape.begin() + segment_ids_shape_rank, x_shape.end(), std::back_inserter(out_vec));
    return out_vec;
  };

  auto out_vec = calc_shape(num_vec, x_shape);
  (void)copy(out_vec.begin(), out_vec.end(), std::back_inserter(output_shape));
  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr UnsortedSegmentSumInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  /* check segment_ids */
  auto ids_ptr = input_args[kInputIndex1]->BuildType();
  MS_EXCEPTION_IF_NULL(ids_ptr);
  std::set<TypePtr> ids_type_set = {kInt16, kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("segment_ids type", ids_ptr, ids_type_set, prim_name);
  /* check num_segments */
  auto num_ptr = input_args[kInputIndex2]->BuildType();
  MS_EXCEPTION_IF_NULL(num_ptr);
  std::map<std::string, TypePtr> args_num_segments;
  (void)args_num_segments.insert(std::make_pair("num_segments", num_ptr));
  const std::set<TypePtr> num_type_set = {kInt16, kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_num_segments, num_type_set, prim_name);
  /* check input_x */
  auto x_type_ptr = input_args[kInputIndex0]->BuildType();
  MS_EXCEPTION_IF_NULL(x_type_ptr);
  return CheckAndConvertUtils::CheckSubClass("input_x", x_type_ptr, {kTensorType}, prim_name);
}
}  // namespace

MIND_API_OPERATOR_IMPL(UnsortedSegmentSum, BaseOperator);
AbstractBasePtr UnsortedSegmentSumInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kMinInputNum = 2;
  const int64_t kMaxInputNum = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kMinInputNum, primitive->name());
  CheckAndConvertUtils::CheckInputArgs(input_args, kLessEqual, kMaxInputNum, primitive->name());
  auto infer_type = UnsortedSegmentSumInferType(primitive, input_args);
  auto infer_shape = UnsortedSegmentSumInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

REGISTER_HOST_DEPENDS(kNameUnsortedSegmentSum, {2});
REGISTER_PRIMITIVE_EVAL_IMPL(UnsortedSegmentSum, prim::kPrimUnsortedSegmentSum, UnsortedSegmentSumInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
