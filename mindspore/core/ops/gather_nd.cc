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
#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr GatherNdInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  if (IsDynamicRank(input_shape) || IsDynamic(indices_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
  }
  // make a scalar to tensor whose shape is (1,)
  if (indices_shape.size() == 0) {
    indices_shape.emplace_back(1);
  }
  auto input_rank = input_shape.size();
  auto indices_rank = indices_shape.size();
  (void)CheckAndConvertUtils::CheckInteger("Input of indices data", SizeToLong(input_rank), kGreaterEqual,
                                           indices_shape[indices_rank - 1], prim_name);
  std::vector<int64_t> output_shape;
  for (size_t i = 0; i < indices_rank - 1; i++) {
    output_shape.push_back(indices_shape[i]);
  }

  if (indices_rank >= 1 && indices_shape[indices_rank - 1] < 0) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
  }
  for (size_t i = LongToSize(indices_shape[indices_rank - 1]); i < input_rank; ++i) {
    output_shape.push_back(input_shape[i]);
  }

  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr GatherNdInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  std::set<TypePtr> int_types = {kInt32, kInt64};
  auto x_type = input_args[kInputIndex0]->BuildType();
  auto indices_type = input_args[kInputIndex1]->BuildType();
  std::set<TypePtr> check_list(common_valid_types_with_complex);
  (void)check_list.insert(kBool);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("indices", indices_type, int_types, "GatherNd");
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, check_list, "GatherNd");
  return x_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(GatherNd, BaseOperator);
AbstractBasePtr GatherNdInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputNum = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, primitive->name());
  auto infer_type = GatherNdInferType(primitive, input_args);
  auto infer_shape = GatherNdInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGGatherNdInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return GatherNdInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return GatherNdInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return GatherNdInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(GatherNd, prim::kPrimGatherNd, AGGatherNdInfer, false);
}  // namespace ops
}  // namespace mindspore
