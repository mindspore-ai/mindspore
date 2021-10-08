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

#include "LayerNormBetaGammaBackprop.h"
#include <memory>
#include <set>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr LayerNormBetaGammaBackpropInferShape(const PrimitivePtr &primitive,
                                                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
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

AbstractBasePtr LayerNormBetaGammaBackpropInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 4;
  (void)CheckAndConvertUtils::CheckInteger("LayerNormBetaGammaBackprop infer", SizeToLong(input_args.size()),
                                           kGreaterEqual, input_num, primitive->name());
  return abstract::MakeAbstract(LayerNormBetaGammaBackpropInferShape(primitive, input_args),
                                LayerNormBetaGammaBackpropInferType(primitive, input_args));
}
REGISTER_PRIMITIVE_EVAL_IMPL(LayerNormBetaGammaBackprop, prim::kPrimLayerNormBetaGammaBackprop,
                             LayerNormBetaGammaBackpropInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
