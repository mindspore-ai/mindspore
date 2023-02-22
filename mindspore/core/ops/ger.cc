/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include <set>
#include <map>

#include "ops/ger.h"
#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
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
abstract::ShapePtr GerInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto first_input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto second_input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  if (IsDynamicRank(first_input_shape) || IsDynamicRank(second_input_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
  }

  int64_t batch_rank = 0;
  if (primitive->HasAttr(kBatchRank)) {
    auto value_ptr = primitive->GetAttr(kBatchRank);
    batch_rank = GetValue<int64_t>(value_ptr);
  }

  std::vector<int64_t> out_shape;
  if (batch_rank == 0) {
    (void)CheckAndConvertUtils::CheckInteger("x1 rank", SizeToLong(first_input_shape.size()), kEqual, kGerShapeNum1,
                                             prim_name);
    (void)CheckAndConvertUtils::CheckInteger("x2 rank", SizeToLong(second_input_shape.size()), kEqual, kGerShapeNum1,
                                             prim_name);
    out_shape = {first_input_shape[0], second_input_shape[0]};
  } else {
    (void)CheckAndConvertUtils::CheckInteger("x1 rank", SizeToLong(first_input_shape.size()), kGreaterEqual,
                                             kGerShapeNum2, prim_name);
    (void)CheckAndConvertUtils::CheckInteger("x2 rank", SizeToLong(second_input_shape.size()), kGreaterEqual,
                                             kGerShapeNum2, prim_name);
    (void)CheckAndConvertUtils::CheckInteger("two inputs rank", SizeToLong(first_input_shape.size()), kEqual,
                                             SizeToLong(second_input_shape.size()), prim_name);

    out_shape.resize(first_input_shape.size() + 1, 1);
    size_t shape_index = 0;
    for (; shape_index < first_input_shape.size() - 1; shape_index++) {
      (void)CheckAndConvertUtils::CheckInteger("two inputs shape", first_input_shape[shape_index], kEqual,
                                               second_input_shape[shape_index], prim_name);
      out_shape[shape_index] = first_input_shape[shape_index];
    }
    out_shape[shape_index] = first_input_shape[first_input_shape.size() - 1];
    out_shape[shape_index + 1] = second_input_shape[second_input_shape.size() - 1];
  }
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr GerInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x1", input_args[0]->BuildType());
  (void)types.emplace("x2", input_args[1]->BuildType());
  return CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim_name);
}
}  // namespace

MIND_API_OPERATOR_IMPL(Ger, BaseOperator);
AbstractBasePtr GerInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto type = GerInferType(primitive, input_args);
  auto shape = GerInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}

// AG means auto generated
class MIND_API AGGerInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return GerInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return GerInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return GerInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Ger, prim::kPrimGer, AGGerInfer, false);
}  // namespace ops
}  // namespace mindspore
