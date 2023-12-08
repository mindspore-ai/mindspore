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

#include "ops/grad/rms_norm_grad.h"

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
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kRmsNormGradInputNum = 4;

BaseShapePtr RmsNormGradInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto x = input_args[1]->BuildShape();
  MS_EXCEPTION_IF_NULL(x);
  auto x_shape_ptr = x->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(x_shape_ptr);

  auto gamma = input_args[3]->BuildShape();
  MS_EXCEPTION_IF_NULL(gamma);
  auto gamma_shape_ptr = gamma->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(x_shape_ptr);

  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{x_shape_ptr, gamma_shape_ptr});
}

TypePtr RmsNormGradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kBFloat16};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("dy", input_args[kIndex0]->BuildType());
  (void)types.emplace("x", input_args[kIndex1]->BuildType());
  (void)types.emplace("gamma", input_args[kIndex3]->BuildType());
  auto output_type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{output_type, kFloat32});
}
}  // namespace

AbstractBasePtr RmsNormGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  for (auto &input : input_args) {
    MS_EXCEPTION_IF_NULL(input);
  }

  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kRmsNormGradInputNum, prim_name);
  auto types = RmsNormGradInferType(primitive, input_args);
  auto shapes = RmsNormGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}

class MIND_API AGRmsNormGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return RmsNormGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return RmsNormGradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return RmsNormGradInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(RmsNormGrad, prim::kPrimRmsNormGrad, AGRmsNormGradInfer, false);
}  // namespace ops
}  // namespace mindspore
