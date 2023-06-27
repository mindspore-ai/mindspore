/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include "ops/compute_accidental_hits.h"

#include <set>
#include <functional>
#include <map>

#include "ops/op_name.h"
#include "ops/array_ops.h"
#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "ops/primitive_c.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kInput1Dim = 2;
constexpr int64_t kInput2Dim = 1;

abstract::TupleShapePtr ComputeAccidentalHitsInferShape(const PrimitivePtr &primitive,
                                                        const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto input1_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape());
  auto input2_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape());
  auto true_classes_shape = input1_shape[kShape];
  auto sampled_candidates_shape = input2_shape[kShape];

  // support dynamic rank
  if (IsDynamicRank(true_classes_shape) || IsDynamicRank(sampled_candidates_shape)) {
    auto out_shape = std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
    auto dyn_rank_shape = std::vector<abstract::BaseShapePtr>{out_shape, out_shape, out_shape};
    return std::make_shared<abstract::TupleShape>(dyn_rank_shape);
  }

  auto true_classes_rank = SizeToLong(true_classes_shape.size());
  (void)CheckAndConvertUtils::CheckInteger("dim of true_classes", true_classes_rank, kEqual, kInput1Dim, prim_name);
  auto sampled_candidates_rank = SizeToLong(sampled_candidates_shape.size());
  (void)CheckAndConvertUtils::CheckInteger("dim of sampled_candidates", sampled_candidates_rank, kEqual, kInput2Dim,
                                           prim_name);

  ShapeVector max_shape;
  if (!IsDynamicShape(true_classes_shape)) {
    const auto num_true_ptr = primitive->GetAttr("num_true");
    MS_EXCEPTION_IF_NULL(num_true_ptr);
    const auto num_true = GetValue<int64_t>(num_true_ptr);
    const auto shape_dim1 = true_classes_shape[1];
    if (shape_dim1 != num_true) {
      MS_EXCEPTION(ValueError) << "For " << prim_name << ", true_classes shape[1]=" << shape_dim1
                               << " must be equal to num_true= " << num_true;
    }
    max_shape = {true_classes_shape[0] * shape_dim1};
  }

  ShapeVector dyn_shape = {abstract::Shape::kShapeDimAny};
  auto ret_shape = std::make_shared<abstract::Shape>(dyn_shape, max_shape);
  auto dyn_dim_shape = std::vector<abstract::BaseShapePtr>{ret_shape, ret_shape, ret_shape};
  return std::make_shared<abstract::TupleShape>(dyn_dim_shape);
}

TuplePtr ComputeAccidentalHitsInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  const std::set<TypePtr> valid_types = {kInt32, kInt64};
  auto true_classes_type = input_args[0]->BuildType();
  auto sampled_candidates_type = input_args[1]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("true_classes", true_classes_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("sampled_candidates", sampled_candidates_type, valid_types,
                                                   prim_name);

  std::vector<TypePtr> type_tuple;
  type_tuple.push_back(true_classes_type);
  type_tuple.push_back(true_classes_type);
  type_tuple.push_back(kFloat32);
  return std::make_shared<Tuple>(type_tuple);
}
}  // namespace

AbstractBasePtr ComputeAccidentalHitsInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  constexpr int64_t kInputsNum = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infer_type = ComputeAccidentalHitsInferType(primitive, input_args);
  auto infer_shape = ComputeAccidentalHitsInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(ComputeAccidentalHits, BaseOperator);

// AG means auto generated
class MIND_API AGComputeAccidentalHitsInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ComputeAccidentalHitsInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ComputeAccidentalHitsInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ComputeAccidentalHitsInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ComputeAccidentalHits, prim::kPrimComputeAccidentalHits, AGComputeAccidentalHitsInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
