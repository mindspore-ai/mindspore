/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "ops/gather_nd.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr GatherNdInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  auto min_shape = shape_map[kMinShape];
  auto max_shape = shape_map[kMaxShape];
  auto input_rank = input_shape.size();
  auto indices_rank = indices_shape.size();
  (void)CheckAndConvertUtils::CheckInteger("Input of indices data", SizeToLong(input_rank), kGreaterEqual,
                                           indices_shape[indices_rank - 1], prim_name);
  std::vector<int64_t> output_shape;
  std::vector<int64_t> min_output_shape;
  std::vector<int64_t> max_output_shape;
  for (size_t i = 0; i < indices_rank - 1; i++) {
    output_shape.push_back(indices_shape[i]);
  }
  for (size_t i = LongToSize(indices_shape[indices_rank - 1]); i < input_rank; ++i) {
    output_shape.push_back(input_shape[i]);
  }
  if (min_shape.empty() || max_shape.empty()) {
    return std::make_shared<abstract::Shape>(output_shape);
  }
  for (size_t i = 0; i < indices_rank - 1; i++) {
    min_output_shape.push_back(indices_shape[i]);
  }
  for (size_t i = LongToSize(indices_shape[indices_rank - 1]); i < input_rank; ++i) {
    min_output_shape.push_back(min_shape[i]);
  }
  for (size_t i = 0; i < indices_rank - 1; i++) {
    max_output_shape.push_back(indices_shape[i]);
  }
  for (size_t i = LongToSize(indices_shape[indices_rank - 1]); i < input_rank; ++i) {
    max_output_shape.push_back(max_shape[i]);
  }
  return std::make_shared<abstract::Shape>(output_shape, min_output_shape, max_output_shape);
}

TypePtr GatherNdInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  if (std::any_of(input_args.begin(), input_args.end(), [](const AbstractBasePtr &arg) { return arg == nullptr; })) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name()
                      << "', the input args userd for infer shape and type, can not be a nullptr.";
  }
  std::set<TypePtr> int_types = {kInt8, kInt16, kInt32, kInt64};
  auto x_type = input_args[kInputIndex0]->BuildType();
  auto indices_type = input_args[kInputIndex1]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("indices", indices_type, int_types, "GatherNd");
  return x_type;
}
}  // namespace

MIND_API_BASE_IMPL(GatherNd, PrimitiveC, BaseOperator);
AbstractBasePtr GatherNdInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t kInputNum = 2;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual, kInputNum,
                                           prim_name);
  auto infer_type = GatherNdInferType(primitive, input_args);
  auto infer_shape = GatherNdInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(GatherNd, prim::kPrimGatherNd, GatherNdInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
