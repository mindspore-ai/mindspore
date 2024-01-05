/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "ops/add_layernorm.h"

#include <memory>
#include <set>
#include <string>
#include <vector>
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/nn_ops.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "utils/shape_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindspore/core/ops/math_ops.h"

namespace mindspore {
namespace ops {
namespace {
BaseShapePtr AddLayerNormInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto x_shape_ptr = input_args[kInputIndex0]->GetShape();
  auto gamma_shape_ptr = input_args[kInputIndex2]->GetShape();
  auto beta_shape_ptr = input_args[kInputIndex3]->GetShape();
  auto x_shape = x_shape_ptr->GetShapeVector();
  auto gamma_shape = gamma_shape_ptr->GetShapeVector();
  auto beta_shape = beta_shape_ptr->GetShapeVector();

  const size_t x_rank = x_shape.size();
  MS_CHECK_VALUE(x_rank != 0,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("x_rank", SizeToLong(x_rank), kNotEqual, 0, primitive));

  MS_CHECK_VALUE(
    !gamma_shape.empty() || !beta_shape.empty(),
    CheckAndConvertUtils::FormatCommMsg("For 'LayerNorm', evaluator gamma or beta can not be an AbstractScalar."));

  if (IsDynamicRank(x_shape) || IsDynamicRank(gamma_shape) || IsDynamicRank(beta_shape)) {
    auto any_shape =
      std::make_shared<abstract::TensorShape>(std::vector<int64_t>{abstract::TensorShape::kShapeRankAny});
    std::vector<BaseShapePtr> shapes_list = {any_shape, any_shape, any_shape, any_shape};
    return std::make_shared<abstract::TupleShape>(shapes_list);
  }

  ShapeVector mean_var_shape = {x_shape[x_rank - 1]};
  std::vector<BaseShapePtr> shapes_list = {x_shape_ptr};
  (void)shapes_list.emplace_back(std::make_shared<abstract::TensorShape>(mean_var_shape));
  (void)shapes_list.emplace_back(std::make_shared<abstract::TensorShape>(mean_var_shape));
  (void)shapes_list.emplace_back(x_shape_ptr);

  return std::make_shared<abstract::TupleShape>(shapes_list);
}

TypePtr AddLayerNormInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto x_type = input_args[kInputIndex0]->GetType();
  auto gamma_type = input_args[kInputIndex2]->GetType();
  auto beta_type = input_args[kInputIndex3]->GetType();
  // the beta and gama shape must be x_shape[begin_params_axis:]

  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  std::vector<TypePtr> types_list;
  bool is_ascend = (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice);
  if (is_ascend) {
    types_list = {x_type, gamma_type, beta_type, x_type};
  } else {
    types_list = {x_type, kFloat32, kFloat32, x_type};
  }

  return std::make_shared<Tuple>(types_list);
}
}  // namespace

MIND_API_OPERATOR_IMPL(AddLayerNorm, BaseOperator);
AbstractBasePtr AddLayerNormInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args) {
  auto output_type = AddLayerNormInferType(primitive, input_args);
  auto output_shape = AddLayerNormInferShape(primitive, input_args);
  return abstract::MakeAbstract(output_shape, output_type);
}

// AG means auto generated
class MIND_API AGAddLayerNormInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return AddLayerNormInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return AddLayerNormInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return AddLayerNormInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(AddLayerNorm, prim::kPrimAddLayerNorm, AGAddLayerNormInfer, false);

}  // namespace ops
}  // namespace mindspore
