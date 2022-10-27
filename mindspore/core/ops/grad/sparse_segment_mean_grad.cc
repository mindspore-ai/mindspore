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

#include "ops/grad/sparse_segment_mean_grad.h"
#include "abstract/dshape.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr SparseSegmentMeanGradInferShape(const PrimitivePtr &prim,
                                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  constexpr size_t kRankNum0 = 0;
  constexpr size_t kRankNum1 = 1;
  constexpr size_t kShapeNum0 = 0;
  constexpr int kDimNum0 = 0;
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto segment_ids_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto output_dim0_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
  if (x_shape.size() < kRankNum1) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', tensor x's rank cannot be less than 1.";
  }
  if (output_dim0_shape.size() != kRankNum0) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', tensor outputdim0 should be a scalar.";
  }
  if (indices_shape[kShapeNum0] != segment_ids_shape[kShapeNum0]) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', tensor indices & segment_ids's ranks mismatch.";
  }
  if (!input_args[kInputIndex3]->BuildValue()->isa<AnyValue>() &&
      !input_args[kInputIndex3]->BuildValue()->isa<None>()) {
    auto output_dim0_value = input_args[kInputIndex3]->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(output_dim0_value);
    auto output_dim0_value_ptr = output_dim0_value->BuildValue();
    MS_EXCEPTION_IF_NULL(output_dim0_value_ptr);
    auto output_dim0_value_ptr_tensor =
      CheckAndConvertUtils::CheckTensorIntValue("output_dim0", output_dim0_value_ptr, prim_name);
    int dim_zero = output_dim0_value_ptr_tensor[kShapeNum0];
    if (dim_zero <= kDimNum0) {
      MS_EXCEPTION(ValueError) << "Input output_dim0 must > 0!";
    } else {
      ShapeVector y_shape = x_shape;
      y_shape[kShapeNum0] = dim_zero;
      return std::make_shared<abstract::Shape>(y_shape);
    }
  } else {
    std::vector<int64_t> output_shape = {-2};
    std::vector<int64_t> min_shape = {1};
    std::vector<int64_t> max_shape = {1};
    return std::make_shared<abstract::Shape>(output_shape, min_shape, max_shape);
  }
}

TypePtr SparseSegmentMeanGradInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto x_type = input_args[kInputIndex0]->BuildType();
  auto indices_type = input_args[kInputIndex1]->BuildType();
  auto segment_ids_type = input_args[kInputIndex2]->BuildType();
  auto output_dim0_type = input_args[kInputIndex3]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, {kFloat32, kFloat64}, prim->name());
  std::map<std::string, TypePtr> types;
  (void)types.emplace("indices", indices_type);
  (void)types.emplace("segment_ids", segment_ids_type);
  (void)types.emplace("output_dim0", output_dim0_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, {kInt32}, prim->name());
  return input_args[kInputIndex0]->BuildType();
}
}  // namespace

MIND_API_OPERATOR_IMPL(SparseSegmentMeanGrad, BaseOperator);
AbstractBasePtr SparseSegmentMeanGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &prim,
                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  const int64_t input_num = static_cast<int64_t>(kInputIndex4);
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
  auto types = SparseSegmentMeanGradInferType(prim, input_args);
  auto shapes = SparseSegmentMeanGradInferShape(prim, input_args);
  return abstract::MakeAbstract(shapes, types);
}
REGISTER_HOST_DEPENDS(kNameSparseSegmentMeanGrad, {3});
REGISTER_PRIMITIVE_EVAL_IMPL(SparseSegmentMeanGrad, prim::kPrimSparseSegmentMeanGrad, SparseSegmentMeanGradInfer,
                             nullptr, true);
}  // namespace ops
}  // namespace mindspore
