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

#include "ops/layer_norm_x_backprop_v2.h"

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
#include "ops/core_ops.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr LayerNormXBackpropV2InferShape(const PrimitivePtr &primitive,
                                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto x_shape = CheckAndConvertUtils::GetTensorInputShape(primitive->name(), input_args, 0);
  auto res_for_gamma_shape = CheckAndConvertUtils::GetTensorInputShape(primitive->name(), input_args, 1);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{x_shape, res_for_gamma_shape});
}

TypePtr LayerNormXBackpropV2InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  // check
  std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  auto x_type = input_args[0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, prim_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{x_type, kFloat32});
}
}  // namespace

MIND_API_OPERATOR_IMPL(LayerNormXBackpropV2, BaseOperator);
AbstractBasePtr LayerNormXBackpropV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 5;
  (void)CheckAndConvertUtils::CheckInteger("LayerNormXBackpropV2 infer", SizeToLong(input_args.size()), kGreaterEqual,
                                           input_num, primitive->name());
  return abstract::MakeAbstract(LayerNormXBackpropV2InferShape(primitive, input_args),
                                LayerNormXBackpropV2InferType(primitive, input_args));
}

// AG means auto generated
class MIND_API AGLayerNormXBackpropV2Infer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return LayerNormXBackpropV2InferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return LayerNormXBackpropV2InferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return LayerNormXBackpropV2Infer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(LayerNormXBackpropV2, prim::kPrimLayerNormXBackpropV2, AGLayerNormXBackpropV2Infer,
                                 false);
}  // namespace ops
}  // namespace mindspore
