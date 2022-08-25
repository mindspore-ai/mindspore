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
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <iostream>

#include "ops/sparse_segment_sqrt_n.h"
#include "abstract/dshape.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr SparseSegmentSqrtNInferShape(const PrimitivePtr &prim,
                                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  constexpr size_t kRankOne = 1;
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto segment_ids_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  if (indices_shape.size() != kRankOne) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", rank of indices should be 1.";
  }
  if (segment_ids_shape.size() != kRankOne) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", rank of segment_ids should be 1.";
  }
  if (x_shape.size() < kRankOne) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', x's rank is less than 1.";
  }
  if (indices_shape[kInputIndex0] != segment_ids_shape[kInputIndex0]) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', ranks of indices and segment_ids mismatch.";
  }
  if (!input_args[kInputIndex2]->BuildValue()->isa<AnyValue>() &&
      !input_args[kInputIndex2]->BuildValue()->isa<None>()) {
    auto segment_ids_value_ptr = input_args[kInputIndex2]->BuildValue();
    MS_EXCEPTION_IF_NULL(segment_ids_value_ptr);
    auto segment_ids_value_ptr_tensor =
      CheckAndConvertUtils::CheckTensorIntValue("segment_ids", segment_ids_value_ptr, prim->name());
    size_t dim_zero = static_cast<size_t>(segment_ids_value_ptr_tensor.back()) + kRankOne;
    if (dim_zero < kRankOne) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', segment_ids must >= 0!";
    } else {
      ShapeVector y_shape = x_shape;
      y_shape[kInputIndex0] = static_cast<int64_t>(dim_zero);
      return std::make_shared<abstract::Shape>(y_shape);
    }
  } else {
    std::vector<int64_t> output_shape = {-2};
    return std::make_shared<abstract::Shape>(output_shape);
  }
}

TypePtr SparseSegmentSqrtNInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto x_type = input_args[kInputIndex0]->BuildType();
  auto indices_type = input_args[kInputIndex1]->BuildType();
  auto segment_ids_type = input_args[kInputIndex2]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  const std::set<TypePtr> common_valid_types = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("indices", indices_type, common_valid_types, prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("segment_ids", segment_ids_type, common_valid_types, prim->name());
  return input_args[kInputIndex0]->BuildType();
}
}  // namespace

MIND_API_OPERATOR_IMPL(SparseSegmentSqrtN, BaseOperator);
AbstractBasePtr SparseSegmentSqrtNInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &prim,
                                        const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  const int64_t input_num = static_cast<int64_t>(kInputIndex3);
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
  auto types = SparseSegmentSqrtNInferType(prim, input_args);
  auto shapes = SparseSegmentSqrtNInferShape(prim, input_args);
  return abstract::MakeAbstract(shapes, types);
}
REGISTER_HOST_DEPENDS(kNameSparseSegmentSqrtN, {2});
REGISTER_PRIMITIVE_EVAL_IMPL(SparseSegmentSqrtN, prim::kPrimSparseSegmentSqrtN, SparseSegmentSqrtNInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
