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

#include "ops/grad/trace_grad.h"

#include <set>
#include <vector>
#include <memory>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ops/core_ops.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
const int64_t kTraceInputsNum = 2;

abstract::ShapePtr TraceGradInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, kTraceInputsNum,
                                     prim_name);
  auto shape_arg = input_args[1];
  MS_EXCEPTION_IF_NULL(shape_arg);
  auto output_shape = GetShapeValue(primitive, shape_arg);
  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr TraceGradInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  const std::set<TypePtr> valid_grad_types = {kInt8,  kInt16,  kInt32,  kInt64,  kFloat16,   kFloat32,   kFloat64,
                                              kUInt8, kUInt16, kUInt32, kUInt64, kComplex64, kComplex128};
  const std::set<TypePtr> valid_shape_types = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTypeValid("x_shape", input_args[1]->BuildType(), valid_shape_types, prim->name());
  return CheckAndConvertUtils::CheckTypeValid("y_grad", input_args[0]->BuildType(), valid_grad_types, prim->name());
}
}  // namespace

MIND_API_OPERATOR_IMPL(TraceGrad, BaseOperator);
AbstractBasePtr TraceGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = TraceGradInferType(primitive, input_args);
  auto infer_shape = TraceGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGTraceGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return TraceGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return TraceGradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return TraceGradInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(TraceGrad, prim::kPrimTraceGrad, AGTraceGradInfer, false);
}  // namespace ops
}  // namespace mindspore
