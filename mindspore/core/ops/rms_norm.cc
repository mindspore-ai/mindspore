/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "ops/rms_norm.h"

#include <memory>
#include <set>
#include <vector>
#include <string>
#include <map>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/nn_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kRmsNormInputNum = 2;

BaseShapePtr RmsNormInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto rstd_shape = x_shape;
  rstd_shape[x_shape.size() - 1] = 1;
  auto y_output_ptr = std::make_shared<abstract::Shape>(x_shape);
  auto rstd_output_ptr = std::make_shared<abstract::Shape>(rstd_shape);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{y_output_ptr, rstd_output_ptr});
}

TypePtr RmsNormInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kBFloat16};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[0]->BuildType());
  (void)types.emplace("gamma", input_args[1]->BuildType());
  auto output_type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{output_type, output_type});
}
}  // namespace

AbstractBasePtr RmsNormInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  for (auto &input : input_args) {
    MS_EXCEPTION_IF_NULL(input);
  }
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kRmsNormInputNum, prim_name);
  auto types = RmsNormInferType(primitive, input_args);
  auto shapes = RmsNormInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}

class MIND_API AGRmsNormInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return RmsNormInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return RmsNormInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return RmsNormInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(RmsNorm, prim::kPrimRmsNorm, AGRmsNormInfer, false);
}  // namespace ops
}  // namespace mindspore
