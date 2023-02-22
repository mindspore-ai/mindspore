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

#include "ops/layer_norm_beta_gamma_backprop_v2.h"

#include <memory>
#include <set>

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
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr LayerNormBetaGammaBackpropV2InferShape(const PrimitivePtr &primitive) {
  MS_EXCEPTION_IF_NULL(primitive);
  ValuePtr gamma_value_ptr = primitive->GetAttr(kShapeGamma);
  MS_EXCEPTION_IF_NULL(gamma_value_ptr);
  auto gamma_shape = GetValue<ShapeVector>(gamma_value_ptr);
  auto gamma_shape_ptr = std::make_shared<abstract::Shape>(gamma_shape);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{gamma_shape_ptr, gamma_shape_ptr});
}

TypePtr LayerNormBetaGammaBackpropV2InferType(const PrimitivePtr &prim,
                                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  auto output_type =
    CheckAndConvertUtils::CheckTensorTypeValid("input_x", input_args[0]->BuildType(), valid_types, prim_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{output_type, output_type});
}
}  // namespace

MIND_API_OPERATOR_IMPL(LayerNormBetaGammaBackpropV2, BaseOperator);
void LayerNormBetaGammaBackpropV2::Init(const std::vector<int64_t> &shape_gamma) { set_shape_gamma(shape_gamma); }

void LayerNormBetaGammaBackpropV2::set_shape_gamma(const std::vector<int64_t> &shape_gamma) {
  (void)AddAttr(kShapeGamma,
                api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kShapeGamma, shape_gamma, name())));
}

std::vector<int64_t> LayerNormBetaGammaBackpropV2::get_shape_gamma() const {
  auto value_ptr = GetAttr(kShapeGamma);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

AbstractBasePtr LayerNormBetaGammaBackpropV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("LayerNormBetaGammaBackpropV2 infer", SizeToLong(input_args.size()),
                                           kGreaterEqual, input_num, primitive->name());
  return abstract::MakeAbstract(LayerNormBetaGammaBackpropV2InferShape(primitive),
                                LayerNormBetaGammaBackpropV2InferType(primitive, input_args));
}

// AG means auto generated
class MIND_API AGLayerNormBetaGammaBackpropV2Infer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return LayerNormBetaGammaBackpropV2InferShape(primitive);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return LayerNormBetaGammaBackpropV2InferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return LayerNormBetaGammaBackpropV2Infer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(LayerNormBetaGammaBackpropV2, prim::kPrimLayerNormBetaGammaBackpropV2,
                                 AGLayerNormBetaGammaBackpropV2Infer, false);
}  // namespace ops
}  // namespace mindspore
