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
#include "ops/ops_func_impl/layer_norm_ext.h"
#include <functional>
#include <string>
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/nn_ops.h"
#include "utils/check_convert_utils.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "utils/ms_context.h"
#include "utils/shape_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
BaseShapePtr LayerNormExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  constexpr int64_t input_num = 5;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);
  auto x_shape_ptr = input_args[kInputIndex0]->GetShape();
  auto gamma_shape_ptr = input_args[kInputIndex2]->GetShape();
  auto beta_shape_ptr = input_args[kInputIndex3]->GetShape();
  const auto &x_shape = x_shape_ptr->GetShapeVector();
  const auto &gamma_shape = gamma_shape_ptr->GetShapeVector();
  const auto &beta_shape = beta_shape_ptr->GetShapeVector();
  auto normalized_shape_opt = GetArrayValue<int64_t>(input_args[kInputIndex1]);

  if (IsDynamicRank(x_shape) || IsDynamicRank(gamma_shape) || IsDynamicRank(beta_shape) ||
      !normalized_shape_opt.has_value() || normalized_shape_opt.value().HasUnknownValue()) {
    auto any_shape =
      std::make_shared<abstract::TensorShape>(std::vector<int64_t>{abstract::TensorShape::kShapeRankAny});
    std::vector<BaseShapePtr> shapes_list = {any_shape, any_shape, any_shape};
    return std::make_shared<abstract::TupleShape>(shapes_list);
  }

  if (normalized_shape_opt.value().ToVector() != gamma_shape && !IsDynamicShape(gamma_shape)) {
    MS_LOG(EXCEPTION) << "For 'LayerNorm', the shape of gamma and beta must be equal to normalized_shape,"
                      << " but got gamma shape: " << gamma_shape << ", beta shape: " << beta_shape
                      << " ,normalized_shape: " << normalized_shape_opt.value().ToVector() << ".";
  }
  const auto norm_dim = gamma_shape.size();
  const auto input_dim = x_shape.size();
  const auto begin_axis = input_dim - norm_dim;
  for (size_t i = begin_axis; i < input_dim; ++i) {
    size_t gamma_beta_shape_dim = i - begin_axis;
    MS_CHECK_VALUE(x_shape[i] <= 0 || ((gamma_shape[gamma_beta_shape_dim] == x_shape[i]) &&
                                       (beta_shape[gamma_beta_shape_dim] == x_shape[i])),
                   CheckAndConvertUtils::FormatCommMsg(
                     "For 'LayerNorm', gamma or beta shape must match input shape, but got input shape: ", x_shape,
                     ", gamma shape: ", gamma_shape, ", beta shape: ", beta_shape, "."));
  }

  ShapeVector mean_shape;
  for (size_t i = 0; i < begin_axis; ++i) {
    (void)mean_shape.emplace_back(x_shape[i]);
  }
  for (size_t i = begin_axis; i < input_dim; ++i) {
    (void)mean_shape.emplace_back(1);
  }
  ShapeVector mean_out_shape = mean_shape;
  ShapeVector rstd_out_shape = mean_shape;

  std::vector<BaseShapePtr> shapes_list = {x_shape_ptr};
  (void)shapes_list.emplace_back(std::make_shared<abstract::Shape>(mean_out_shape));
  (void)shapes_list.emplace_back(std::make_shared<abstract::Shape>(rstd_out_shape));
  return std::make_shared<abstract::TupleShape>(shapes_list);
}

TypePtr LayerNormExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) const {
  // outputs: output, mean_out, rstd_out
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string op_name = primitive->name();
  constexpr int64_t input_num = 5;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);

  auto input_type = input_args[kInputIndex0]->GetType();
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  std::vector<TypePtr> types_list;
  types_list = {input_type, input_type, input_type};
  return std::make_shared<Tuple>(types_list);
}

TypePtrList LayerNormExtFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  return {x_tensor->Dtype(), x_tensor->Dtype(), x_tensor->Dtype()};
}

ShapeArray LayerNormExtFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  const auto &gamma_tensor = input_values[kInputIndex2]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  const auto &x_shape = x_tensor->shape();
  const auto &gamma_shape = gamma_tensor->shape();

  const auto norm_dim = gamma_shape.size();
  const auto input_dim = x_shape.size();
  const auto begin_axis = input_dim - norm_dim;

  ShapeVector mean_shape;
  for (size_t i = 0; i < begin_axis; ++i) {
    (void)mean_shape.emplace_back(x_shape[i]);
  }
  for (size_t i = begin_axis; i < input_dim; ++i) {
    (void)mean_shape.emplace_back(1);
  }
  ShapeVector mean_out_shape = mean_shape;
  ShapeVector rstd_out_shape = mean_shape;

  return {x_shape, mean_out_shape, rstd_out_shape};
}

REGISTER_SIMPLE_INFER(kNameLayerNormExt, LayerNormExtFuncImpl)
}  // namespace ops
}  // namespace mindspore
