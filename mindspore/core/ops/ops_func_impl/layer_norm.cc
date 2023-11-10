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

#include "ops/ops_func_impl/layer_norm.h"

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

int64_t CheckAxis(const std::string &op, const std::string &args_name, int64_t axis_value, int64_t minimum, int64_t max,
                  const std::string &rank_name) {
  if (axis_value >= max || axis_value < minimum) {
    MS_LOG(EXCEPTION) << "For primitive[" << op << "], " << rank_name << "'s rank is " << max << ", while the "
                      << "\'" << args_name << "\' value should be in the range [" << minimum << ", " << max
                      << "), but got " << axis_value;
  }
  if (axis_value < 0) {
    axis_value = axis_value + max;
  }
  return axis_value;
}

}  // namespace

BaseShapePtr LayerNormFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto x_shape_ptr = input_args[kInputIndex0]->GetShape();
  auto gamma_shape_ptr = input_args[kInputIndex1]->GetShape();
  auto beta_shape_ptr = input_args[kInputIndex2]->GetShape();
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
    std::vector<BaseShapePtr> shapes_list = {any_shape, any_shape, any_shape};
    return std::make_shared<abstract::TupleShape>(shapes_list);
  }

  // begin_norm_axis and begin_params_axis must be smaller than the size of input_x and >= -1
  ValuePtr bna_ptr = input_args[kInputIndex3]->GetValue();
  std::optional<int64_t> bna_opt = GetScalarValue<int64_t>(bna_ptr);
  if (!bna_opt.has_value()) {
    auto any_shape =
      std::make_shared<abstract::TensorShape>(std::vector<int64_t>{abstract::TensorShape::kShapeRankAny});
    std::vector<BaseShapePtr> shapes_list = {x_shape_ptr, any_shape, any_shape};
    return std::make_shared<abstract::TupleShape>(shapes_list);
  }
  int64_t bna_value = bna_opt.value();
  int64_t begin_norm_axis = CheckAxis(op_name, "begin_norm_axis", bna_value, -1, SizeToLong(x_rank), "input_x");

  std::vector<BaseShapePtr> shapes_list = {x_shape_ptr};
  auto mean_var_shape = CalLayerNormMeanAndVarShape(begin_norm_axis, x_shape);
  (void)shapes_list.emplace_back(std::make_shared<abstract::TensorShape>(mean_var_shape));
  (void)shapes_list.emplace_back(std::make_shared<abstract::TensorShape>(mean_var_shape));

  ValuePtr bpa_ptr = input_args[kInputIndex4]->GetValue();
  std::optional<int64_t> bpa_opt = GetScalarValue<int64_t>(bpa_ptr);
  if (!bpa_opt.has_value()) {
    return std::make_shared<abstract::TupleShape>(shapes_list);
  }
  int64_t bpa_value = bpa_opt.value();
  int64_t begin_params_axis = CheckAxis(op_name, "begin_params_axis", bpa_value, -1, SizeToLong(x_rank), "input_x");

  size_t begin_params_axis_u = LongToSize(begin_params_axis);
  MS_CHECK_VALUE((begin_params_axis_u <= x_rank) && (gamma_shape.size() + begin_params_axis_u >= x_rank) &&
                   (beta_shape.size() + begin_params_axis_u >= x_rank),
                 CheckAndConvertUtils::FormatCommMsg(
                   "For '", op_name,
                   "', begin_params_axis must be less than or equal to input_x shape size, gamma shape size add "
                   "begin_params_axis must be equal to or greater than input_x shape size, and beta shape size add "
                   "begin_params_axis must be equal to or greater than input_x shape size, But got begin_params_axis: ",
                   begin_params_axis_u, ", input_x shape size: ", x_rank, ", gamma shape size: ", gamma_shape.size(),
                   ", beta shape size: ", beta_shape.size(), "."));

  for (size_t i = begin_params_axis_u; i < x_rank; ++i) {
    size_t gamma_beta_shape_dim = i - begin_params_axis_u;
    MS_CHECK_VALUE(x_shape[i] <= 0 || ((gamma_shape[gamma_beta_shape_dim] == x_shape[i]) &&
                                       (beta_shape[gamma_beta_shape_dim] == x_shape[i])),
                   CheckAndConvertUtils::FormatCommMsg(
                     "For '", op_name, "', gamma or beta shape must match input shape, but got input shape: ", x_shape,
                     ", gamma shape: ", gamma_shape, ", beta shape: ", beta_shape, "."));
  }

  return std::make_shared<abstract::TupleShape>(shapes_list);
}

TypePtr LayerNormFuncImpl::InferType(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  auto x_type = input_args[kInputIndex0]->GetType();
  // the beta and gama shape must be x_shape[begin_params_axis:]

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
}  // namespace ops
}  // namespace mindspore
