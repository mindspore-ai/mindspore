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

#include <set>
#include <vector>
#include <memory>

#include "ops/diag_part.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr DiagPartInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  constexpr size_t kScaleNum = 2;
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->GetShapeTrack())[kShape];
  if (IsDynamicRank(input_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
  }
  if ((input_shape.size() % kScaleNum) != 0 || input_shape.size() == 0) {
    MS_EXCEPTION(ValueError) << "For 'DiagPart', input rank must be non-zero and even, but got rank: "
                             << input_shape.size() << ".";
  }
  auto length = input_shape.size() / kScaleNum;
  std::vector<int64_t> out_shape;
  for (size_t i = 0; i < length; i++) {
    if (input_shape[i + length] > 0 && input_shape[i] > 0) {
      CheckAndConvertUtils::Check("input_shape[i + rank(input_shape) / 2]", input_shape[i + length], kEqual,
                                  input_shape[i], op_name, ValueError);
    }
    (void)out_shape.emplace_back(input_shape[i]);
  }
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr DiagPartInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto x_dtype = input_args[0]->BuildType();
  const std::set<TypePtr> valid_types = {kInt32, kInt64, kFloat16, kFloat32, kFloat64, kComplex64, kComplex128};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input", x_dtype, valid_types, primitive->name());
  return x_dtype;
}
}  // namespace

MIND_API_OPERATOR_IMPL(DiagPart, BaseOperator);
AbstractBasePtr DiagPartInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, 1, primitive->name());
  auto infer_type = DiagPartInferType(primitive, input_args);
  auto infer_shape = DiagPartInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGDiagPartInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return DiagPartInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return DiagPartInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return DiagPartInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(DiagPart, prim::kPrimDiagPart, AGDiagPartInfer, false);
}  // namespace ops
}  // namespace mindspore
