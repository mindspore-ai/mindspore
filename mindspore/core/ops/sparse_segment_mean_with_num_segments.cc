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

#include <algorithm>

#include "ops/sparse_segment_mean_with_num_segments.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr SparseSegmentMeanWithNumSegmentsInferShape(const PrimitivePtr &prim,
                                                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  constexpr size_t kRankOne = 1;
  constexpr size_t kDimOne = 1;
  constexpr size_t kShapeZero = 0;
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  if (IsDynamicRank(x_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
  }
  auto indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto segment_ids_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto num_segments_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
  if (!IsDynamicRank(indices_shape) && !IsDynamicRank(segment_ids_shape) && !IsDynamicRank(num_segments_shape)) {
    if (indices_shape.size() != kRankOne) {
      MS_EXCEPTION(ValueError) << "For " << prim_name << ", rank of indices should be 1.";
    }
    if (segment_ids_shape.size() != kRankOne) {
      MS_EXCEPTION(ValueError) << "For " << prim_name << ", rank of segment_ids should be 1.";
    }
    if (x_shape.size() < kRankOne) {
      MS_EXCEPTION(ValueError) << "For " << prim_name << ", rank of x cannot be less than 1.";
    }
    if (!IsDynamic(indices_shape) && !IsDynamic(segment_ids_shape) &&
        indices_shape[kShapeZero] != segment_ids_shape[kShapeZero]) {
      MS_EXCEPTION(ValueError) << "For " << prim_name << ", indices and segment_ids's ranks mismatch.";
    }
    if (num_segments_shape.size() > kRankOne) {
      MS_EXCEPTION(ValueError) << "For " << prim_name << ", rank of num_segments should be 0 or 1.";
    }
    if (!IsDynamic(num_segments_shape)) {
      if (num_segments_shape.size() == kRankOne && num_segments_shape[kShapeZero] != static_cast<int64_t>(kDimOne)) {
        MS_EXCEPTION(ValueError) << "For " << prim_name << ", the num element of num_segments should be 1.";
      }
    }
  }
  if (input_args[kInputIndex3]->isa<abstract::AbstractTensor>() &&
      input_args[kInputIndex3]->BuildValue()->isa<tensor::Tensor>()) {
    auto num_segments = input_args[kInputIndex3]->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(num_segments);
    auto num_segments_value = num_segments->BuildValue();
    MS_EXCEPTION_IF_NULL(num_segments_value);
    auto num_segments_value_tensor =
      CheckAndConvertUtils::CheckTensorIntValue("num_segments", num_segments_value, prim_name);
    size_t dim_zero = num_segments_value_tensor.back();
    if (dim_zero < kInputIndex1) {
      MS_EXCEPTION(ValueError) << "For " << prim_name
                               << ", num_segments must bigger than the last number of segment_ids, "
                               << "but got " << dim_zero << ".";
    } else {
      ShapeVector y_shape = x_shape;
      y_shape[kInputIndex0] = static_cast<int64_t>(dim_zero);
      return std::make_shared<abstract::Shape>(y_shape);
    }
  } else {
    ShapeVector output_shape = x_shape;
    output_shape[kInputIndex0] = -1;
    return std::make_shared<abstract::Shape>(output_shape);
  }
}

TypePtr SparseSegmentMeanWithNumSegmentsInferType(const PrimitivePtr &prim,
                                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  auto x_type = input_args[kInputIndex0]->BuildType();
  auto indices_type = input_args[kInputIndex1]->BuildType();
  auto segment_ids_type = input_args[kInputIndex2]->BuildType();
  auto num_segments_type = input_args[kInputIndex3]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  const std::set<TypePtr> common_valid_types = {kInt32, kInt64};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("indices", indices_type);
  (void)types.emplace("segment_ids", segment_ids_type);
  (void)types.emplace("num_segments", num_segments_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, common_valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, prim_name);
  return input_args[kInputIndex0]->BuildType();
}
}  // namespace

AbstractBasePtr SparseSegmentMeanWithNumSegmentsInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &prim,
                                                      const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  const int64_t kInputsNum = 4;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, prim->name());
  auto types = SparseSegmentMeanWithNumSegmentsInferType(prim, input_args);
  auto shapes = SparseSegmentMeanWithNumSegmentsInferShape(prim, input_args);
  return abstract::MakeAbstract(shapes, types);
}

MIND_API_OPERATOR_IMPL(SparseSegmentMeanWithNumSegments, BaseOperator);

// AG means auto generated
class MIND_API AGSparseSegmentMeanWithNumSegmentsInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseSegmentMeanWithNumSegmentsInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseSegmentMeanWithNumSegmentsInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseSegmentMeanWithNumSegmentsInfer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {3}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SparseSegmentMeanWithNumSegments, prim::kPrimSparseSegmentMeanWithNumSegments,
                                 AGSparseSegmentMeanWithNumSegmentsInfer, false);
}  // namespace ops
}  // namespace mindspore
