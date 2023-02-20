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

#include "ops/LayerNormBetaGammaBackprop.h"

#include <memory>
#include <set>
#include <map>
#include <string>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr LayerNormBetaGammaBackpropInferShape(const PrimitivePtr &primitive) {
  MS_EXCEPTION_IF_NULL(primitive);
  ValuePtr gamma_value_ptr = primitive->GetAttr("shape_gamma");
  MS_EXCEPTION_IF_NULL(gamma_value_ptr);
  auto gamma_shape = GetValue<ShapeVector>(gamma_value_ptr);
  auto gamma_shape_ptr = std::make_shared<abstract::Shape>(gamma_shape);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{gamma_shape_ptr, gamma_shape_ptr});
}

TypePtr LayerNormBetaGammaBackpropInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  // check
  std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  std::map<std::string, TypePtr> types;
  const int64_t beta_index = 1;
  const int64_t gamma_index = 2;
  (void)types.emplace("beta", input_args[beta_index]->BuildType());
  (void)types.emplace("gamma", input_args[gamma_index]->BuildType());
  auto output_type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{output_type, output_type});
}
}  // namespace

MIND_API_OPERATOR_IMPL(LayerNormBetaGammaBackprop, BaseOperator);
AbstractBasePtr LayerNormBetaGammaBackpropInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const int64_t input_num = 4;
  (void)CheckAndConvertUtils::CheckInteger("LayerNormBetaGammaBackprop infer", SizeToLong(input_args.size()),
                                           kGreaterEqual, input_num, primitive->name());
  return abstract::MakeAbstract(LayerNormBetaGammaBackpropInferShape(primitive),
                                LayerNormBetaGammaBackpropInferType(primitive, input_args));
}

// AG means auto generated
class MIND_API AGLayerNormBetaGammaBackpropInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return LayerNormBetaGammaBackpropInferShape(primitive);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return LayerNormBetaGammaBackpropInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return LayerNormBetaGammaBackpropInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(LayerNormBetaGammaBackprop, prim::kPrimLayerNormBetaGammaBackprop,
                                 AGLayerNormBetaGammaBackpropInfer, false);
}  // namespace ops
}  // namespace mindspore
