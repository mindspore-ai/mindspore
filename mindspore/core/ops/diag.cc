/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include <vector>
#include <memory>

#include "ops/diag.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr DiagInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->GetShapeTrack())[kShape];
  // Support the dynamic rank.
  if (IsDynamicRank(input_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }
  std::vector<int64_t> out_shape;

  int64_t batch_rank = 0;
  if (primitive->HasAttr(kBatchRank)) {
    auto value_ptr = primitive->GetAttr(kBatchRank);
    batch_rank = GetValue<int64_t>(value_ptr);
  }

  // Vmap
  if (batch_rank > 0) {
    (void)CheckAndConvertUtils::CheckInteger("input rank", SizeToLong(input_shape.size()), kGreaterEqual,
                                             (batch_rank + 1), primitive->name());
    // The shape of batch rank.
    (void)out_shape.insert(out_shape.end(), input_shape.begin(), input_shape.begin() + batch_rank);
    // The shape of op data.
    (void)out_shape.insert(out_shape.end(), input_shape.begin() + batch_rank, input_shape.end());
    (void)out_shape.insert(out_shape.end(), input_shape.begin() + batch_rank, input_shape.end());
  } else {
    (void)CheckAndConvertUtils::CheckInteger("input rank", SizeToLong(input_shape.size()), kGreaterEqual, 1,
                                             primitive->name());
    (void)out_shape.insert(out_shape.end(), input_shape.begin(), input_shape.end());
    (void)out_shape.insert(out_shape.end(), input_shape.begin(), input_shape.end());
  }

  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr DiagInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto x_dtype = input_args[0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input_x", x_dtype, common_valid_types_with_complex,
                                                   primitive->name());
  return x_dtype;
}
}  // namespace

MIND_API_OPERATOR_IMPL(Diag, BaseOperator);
AbstractBasePtr DiagInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, 1, primitive->name());
  auto infer_type = DiagInferType(primitive, input_args);
  auto infer_shape = DiagInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGDiagInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return DiagInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return DiagInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return DiagInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Diag, prim::kPrimDiag, AGDiagInfer, false);
}  // namespace ops
}  // namespace mindspore
