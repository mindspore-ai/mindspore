/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "ops/layer_norm.h"

#include "utils/ms_context.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/param_validator.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
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
ShapeVector CalLayerNormMeanAndVarShape(int64_t begin_norm_axis, const ShapeVector &input_shape) {
  auto mean_var_shape_value = input_shape;
  const size_t input_rank = input_shape.size();
  if (begin_norm_axis == -1) {
    mean_var_shape_value[input_rank - 1] = 1;
  } else {
    for (size_t i = LongToSize(begin_norm_axis); i < input_rank; i++) {
      mean_var_shape_value[i] = 1;
    }
  }
  return mean_var_shape_value;
}
}  // namespace

MIND_API_OPERATOR_IMPL(LayerNorm, BaseOperator);
AbstractBasePtr LayerNormInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  // Inputs: three tensors(x, gamma, beta).
  // outputs: y, mean, variance
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string op_name = primitive->name();
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);
  const int64_t x_index = 0;
  const int64_t gamma_index = 1;
  const int64_t beta_index = 2;
  auto input_x = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, x_index);
  auto gamma = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, gamma_index);
  auto beta = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, beta_index);

  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  std::vector<TypePtr> types_list;
  bool is_ascend = (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice);
  if (is_ascend) {
    types_list = {input_x->BuildType(), input_x->BuildType(), input_x->BuildType()};
  } else {
    types_list = {input_x->BuildType(), kFloat32, kFloat32};
  }

  auto input_shape = input_x->shape();
  MS_EXCEPTION_IF_NULL(input_shape);
  auto gamma_shape = dyn_cast<abstract::Shape>(gamma->BuildShape());
  auto beta_shape = dyn_cast<abstract::Shape>(beta->BuildShape());
  MS_EXCEPTION_IF_NULL(gamma_shape);
  MS_EXCEPTION_IF_NULL(beta_shape);

  auto const &input_shape_list = input_shape->shape();
  auto const &gamma_shape_list = gamma_shape->shape();
  auto const &beta_shape_list = beta_shape->shape();

  if (IsDynamicRank(input_shape_list) || IsDynamicRank(gamma_shape_list) || IsDynamicRank(beta_shape_list)) {
    auto any_shape = std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
    std::vector<BaseShapePtr> shapes_list = {any_shape, any_shape, any_shape};
    return abstract::MakeAbstract(std::make_shared<abstract::TupleShape>(shapes_list),
                                  std::make_shared<Tuple>(types_list));
  }

  const size_t input_rank = input_shape_list.size();
  if (input_rank == 0) {
    MS_LOG(EXCEPTION) << "For '" << op_name << "', input_rank can not be zero, but got: " << input_rank << ".";
  }

  // begin_norm_axis and begin_params_axis must be smaller than the size of input_x and >= -1
  ValuePtr bna_ptr = primitive->GetAttr("begin_norm_axis");
  int64_t begin_norm_axis =
    abstract::CheckAxis(op_name, "begin_norm_axis", bna_ptr, -1, SizeToLong(input_rank), "input_x");
  ValuePtr bpa_ptr = primitive->GetAttr("begin_params_axis");
  int64_t begin_params_axis =
    abstract::CheckAxis(op_name, "begin_params_axis", bpa_ptr, -1, SizeToLong(input_rank), "input_x");

  // the beta and gama shape must be x_shape[begin_params_axis:]
  auto valid_types = {kFloat16, kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x_dtype", input_args[x_index]->BuildType(), valid_types, op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("gamma_dtype", input_args[gamma_index]->BuildType(), valid_types,
                                                   op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("beta_dtype", input_args[beta_index]->BuildType(), valid_types,
                                                   op_name);

  if (gamma_shape_list.empty() || beta_shape_list.empty()) {
    MS_EXCEPTION(ValueError) << "For 'LayerNorm', evaluator gamma or beta can not be an AbstractScalar.";
  }

  size_t begin_params_axis_u = LongToSize(begin_params_axis);
  if ((begin_params_axis_u > input_shape_list.size()) ||
      (gamma_shape_list.size() + begin_params_axis_u < input_shape_list.size()) ||
      (beta_shape_list.size() + begin_params_axis_u < input_shape_list.size())) {
    MS_EXCEPTION(ValueError)
      << "For '" << op_name
      << "', begin_params_axis must be less than or equal to input_x shape size, gamma shape size add "
         "begin_params_axis must be equal to or greater than input_x shape size, and beta shape size add "
         "begin_params_axis must be equal to or greater than input_x shape size, But got begin_params_axis: "
      << begin_params_axis_u << ", input_x shape size: " << input_shape_list.size()
      << ", gamma shape size: " << gamma_shape_list.size() << ", beta shape size: " << beta_shape_list.size() << ".";
  }
  for (size_t i = begin_params_axis_u; i < input_shape_list.size(); ++i) {
    size_t gamma_beta_shape_dim = i - begin_params_axis_u;
    if (input_shape_list[i] > 0 && ((gamma_shape_list[gamma_beta_shape_dim] != input_shape_list[i]) ||
                                    (beta_shape_list[gamma_beta_shape_dim] != input_shape_list[i]))) {
      MS_EXCEPTION(ValueError) << "For '" << op_name
                               << "', gamma or beta shape must match input shape, but got input shape: "
                               << input_shape->ToString() << ", gamma shape: " << gamma_shape->ToString()
                               << ", beta shape: " << beta_shape->ToString() << ".";
    }
  }

  std::vector<BaseShapePtr> shapes_list = {input_x->BuildShape()};
  auto mean_var_shape = CalLayerNormMeanAndVarShape(begin_norm_axis, input_shape->shape());
  (void)shapes_list.emplace_back(std::make_shared<abstract::Shape>(mean_var_shape));
  (void)shapes_list.emplace_back(std::make_shared<abstract::Shape>(mean_var_shape));

  return abstract::MakeAbstract(std::make_shared<abstract::TupleShape>(shapes_list),
                                std::make_shared<Tuple>(types_list));
}

void LayerNorm::Init(const int64_t begin_norm_axis, const int64_t begin_params_axis, const float epsilon) {
  this->set_begin_norm_axis(begin_norm_axis);
  this->set_begin_params_axis(begin_params_axis);
  this->set_epsilon(epsilon);
}
void LayerNorm::set_begin_norm_axis(const int64_t begin_norm_axis) {
  (void)this->AddAttr(kBeginNormAxis, api::MakeValue(begin_norm_axis));
}
void LayerNorm::set_begin_params_axis(const int64_t begin_params_axis) {
  (void)this->AddAttr(kBeginParamsAxis, api::MakeValue(begin_params_axis));
}
void LayerNorm::set_epsilon(const float epsilon) { (void)this->AddAttr(kEpsilon, api::MakeValue(epsilon)); }

int64_t LayerNorm::get_begin_norm_axis() const {
  auto value_ptr = this->GetAttr(kBeginNormAxis);
  return GetValue<int64_t>(value_ptr);
}
int64_t LayerNorm::get_begin_params_axis() const {
  auto value_ptr = this->GetAttr(kBeginParamsAxis);
  return GetValue<int64_t>(value_ptr);
}
float LayerNorm::get_epsilon() const {
  auto value_ptr = this->GetAttr(kEpsilon);
  return GetValue<float>(value_ptr);
}
REGISTER_PRIMITIVE_EVAL_IMPL(LayerNorm, prim::kPrimLayerNorm, LayerNormInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
