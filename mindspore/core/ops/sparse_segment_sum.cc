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

#include "ops/sparse_segment_sum.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr SparseSegmentSumInferShape(const PrimitivePtr &prim,
                                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto segment_ids_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("indices_shape", SizeToLong(indices_shape.size()), kEqual,
                                           SizeToLong(kInputIndex1), prim->name());
  (void)CheckAndConvertUtils::CheckInteger("segment_ids_shape", SizeToLong(segment_ids_shape.size()), kEqual,
                                           SizeToLong(kInputIndex1), prim->name());
  if (x_shape.size() < kInputIndex1) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', "
                             << "x's rank must be greater than 1, but got [" << x_shape.size() << "].";
  }
  if (indices_shape[kInputIndex0] != segment_ids_shape[kInputIndex0]) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the rank of indices and segment_ids should be the same, "
                             << "but got indices [" << indices_shape[kInputIndex0] << "] "
                             << "and segment_ids [" << segment_ids_shape[kInputIndex0] << "].";
  }
  if (!input_args[kInputIndex2]->BuildValue()->isa<AnyValue>() &&
      !input_args[kInputIndex2]->BuildValue()->isa<None>()) {
    auto segment_ids_value_ptr = input_args[kInputIndex2]->BuildValue();
    MS_EXCEPTION_IF_NULL(segment_ids_value_ptr);
    auto segment_ids_value_ptr_tensor =
      CheckAndConvertUtils::CheckTensorIntValue("segment_ids", segment_ids_value_ptr, prim->name());
    int64_t dim_zero = segment_ids_value_ptr_tensor.back() + kInputIndex1;
    if (dim_zero < SizeToLong(kInputIndex1)) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', segment_ids must be greater or equal to 0, "
                               << "but got [" << dim_zero << "].";
    } else {
      ShapeVector y_shape = x_shape;
      y_shape[kInputIndex0] = dim_zero;
      return std::make_shared<abstract::Shape>(y_shape);
    }
  } else {
    std::vector<int64_t> output_shape = {-2};
    std::vector<int64_t> min_shape = {1};
    std::vector<int64_t> max_shape = {1};
    return std::make_shared<abstract::Shape>(output_shape, min_shape, max_shape);
  }
}

TypePtr SparseSegmentSumInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  auto x_type = input_args[kInputIndex0]->BuildType();
  auto indices_type = input_args[kInputIndex1]->BuildType();
  auto segment_ids_type = input_args[kInputIndex2]->BuildType();
  const std::set<TypePtr> valid_types = {kInt8, kInt16, kInt32, kInt64, kUInt8, kUInt16, kFloat16, kFloat32, kFloat64};
  const std::set<TypePtr> common_valid_types = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, prim_name);
  std::map<std::string, TypePtr> types;
  (void)types.emplace("indices", indices_type);
  (void)types.emplace("segment_ids", segment_ids_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, common_valid_types, prim->name());
  return input_args[kInputIndex0]->BuildType();
}
}  // namespace

MIND_API_OPERATOR_IMPL(SparseSegmentSum, BaseOperator);
AbstractBasePtr SparseSegmentSumInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &prim,
                                      const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  const int64_t input_num = static_cast<int64_t>(kInputIndex3);
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
  auto types = SparseSegmentSumInferType(prim, input_args);
  auto shapes = SparseSegmentSumInferShape(prim, input_args);
  return abstract::MakeAbstract(shapes, types);
}
REGISTER_HOST_DEPENDS(kNameSparseSegmentSum, {2});
REGISTER_PRIMITIVE_EVAL_IMPL(SparseSegmentSum, prim::kPrimSparseSegmentSum, SparseSegmentSumInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
