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
class MIND_API LayerNormInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto op_name = primitive->name();
    const int64_t input_num = 3;
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);
    auto x_shape_ptr = input_args[kInputIndex0]->BuildShape();
    auto gamma_shape_ptr = input_args[kInputIndex1]->BuildShape();
    auto beta_shape_ptr = input_args[kInputIndex2]->BuildShape();
    auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x_shape_ptr)[kShape];
    auto gamma_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(gamma_shape_ptr)[kShape];
    auto beta_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(beta_shape_ptr)[kShape];
    const size_t x_rank = x_shape.size();
    if (x_rank == 0) {
      MS_LOG(EXCEPTION) << "For '" << op_name << "', input_rank can not be zero, but got: " << x_rank << ".";
    }
    if (gamma_shape.empty() || beta_shape.empty()) {
      MS_EXCEPTION(ValueError) << "For 'LayerNorm', the gamma and beta should be at least 1-dimensional, i.e., "
                                  "containing at least one element, but got gamma shape: "
                               << gamma_shape << ", beta shape: " << beta_shape << ".";
    }

    if (IsDynamicRank(x_shape) || IsDynamicRank(gamma_shape) || IsDynamicRank(beta_shape)) {
      auto any_shape = std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
      std::vector<BaseShapePtr> shapes_list = {any_shape, any_shape, any_shape};
      return std::make_shared<abstract::TupleShape>(shapes_list);
    }

    // begin_norm_axis and begin_params_axis must be smaller than the size of input_x and >= -1
    ValuePtr bna_ptr = primitive->GetAttr("begin_norm_axis");
    int64_t begin_norm_axis =
      abstract::CheckAxis(op_name, "begin_norm_axis", bna_ptr, -1, SizeToLong(x_rank), "input_x");
    ValuePtr bpa_ptr = primitive->GetAttr("begin_params_axis");
    int64_t begin_params_axis =
      abstract::CheckAxis(op_name, "begin_params_axis", bpa_ptr, -1, SizeToLong(x_rank), "input_x");

    size_t begin_params_axis_u = LongToSize(begin_params_axis);
    if ((begin_params_axis_u > x_rank) || (gamma_shape.size() + begin_params_axis_u < x_rank) ||
        (beta_shape.size() + begin_params_axis_u < x_rank)) {
      MS_EXCEPTION(ValueError)
        << "For '" << op_name
        << "', begin_params_axis must be less than or equal to input_x shape size, gamma shape size add "
           "begin_params_axis must be equal to or greater than input_x shape size, and beta shape size add "
           "begin_params_axis must be equal to or greater than input_x shape size, But got begin_params_axis: "
        << begin_params_axis_u << ", input_x shape size: " << x_rank << ", gamma shape size: " << gamma_shape.size()
        << ", beta shape size: " << beta_shape.size() << ".";
    }
    for (size_t i = begin_params_axis_u; i < x_rank; ++i) {
      size_t gamma_beta_shape_dim = i - begin_params_axis_u;
      if (x_shape[i] > 0 &&
          ((gamma_shape[gamma_beta_shape_dim] != x_shape[i]) || (beta_shape[gamma_beta_shape_dim] != x_shape[i]))) {
        MS_EXCEPTION(ValueError) << "For '" << op_name
                                 << "', gamma or beta shape must match input shape, but got input shape: " << x_shape
                                 << ", gamma shape: " << gamma_shape << ", beta shape: " << beta_shape << ".";
      }
    }

    std::vector<BaseShapePtr> shapes_list = {x_shape_ptr};
    auto mean_var_shape = CalLayerNormMeanAndVarShape(begin_norm_axis, x_shape);
    (void)shapes_list.emplace_back(std::make_shared<abstract::Shape>(mean_var_shape));
    (void)shapes_list.emplace_back(std::make_shared<abstract::Shape>(mean_var_shape));

    return std::make_shared<abstract::TupleShape>(shapes_list);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    // Inputs: three tensors(x, gamma, beta).
    // outputs: y, mean, variance
    MS_EXCEPTION_IF_NULL(primitive);
    const std::string op_name = primitive->name();
    const int64_t input_num = 3;
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);

    auto x_type = input_args[kInputIndex0]->BuildType();
    auto gamma_type = input_args[kInputIndex1]->BuildType();
    auto beta_type = input_args[kInputIndex2]->BuildType();
    // the beta and gama shape must be x_shape[begin_params_axis:]
    auto valid_types = {kBFloat16, kFloat16, kFloat32, kFloat64};
    (void)CheckAndConvertUtils::CheckTensorTypeValid("x_dtype", x_type, valid_types, op_name);
    (void)CheckAndConvertUtils::CheckTensorTypeValid("gamma_dtype", gamma_type, valid_types, op_name);
    (void)CheckAndConvertUtils::CheckTensorTypeValid("beta_dtype", beta_type, valid_types, op_name);

    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    std::vector<TypePtr> types_list;
    bool is_ascend = (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice);
    if (is_ascend) {
      types_list = {x_type, x_type, x_type};
    } else {
      types_list = {x_type, kFloat32, kFloat32};
    }

    return std::make_shared<Tuple>(types_list);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(LayerNorm, prim::kPrimLayerNorm, LayerNormInfer, false);

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
}  // namespace ops
}  // namespace mindspore
