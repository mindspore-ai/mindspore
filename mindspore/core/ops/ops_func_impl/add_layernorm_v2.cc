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

#include "ops/ops_func_impl/add_layernorm_v2.h"
#include <complex>
#include <memory>
#include <vector>
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
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
#include "mindspore/core/ops/math_ops.h"

namespace mindspore {
namespace ops {
using float_complex = std::complex<float>;
using double_complex = std::complex<double>;

abstract::BaseShapePtr AddLayerNormV2FuncImpl::InferShape(const PrimitivePtr &prim,
                                                          const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(prim);
  auto x_shape_ptr = input_args[kInputIndex0]->GetShape();
  auto gamma_shape_ptr = input_args[kInputIndex2]->GetShape();
  auto beta_shape_ptr = input_args[kInputIndex3]->GetShape();
  MS_EXCEPTION_IF_NULL(x_shape_ptr);
  MS_EXCEPTION_IF_NULL(gamma_shape_ptr);
  MS_EXCEPTION_IF_NULL(beta_shape_ptr);

  auto x_shape = x_shape_ptr->GetShapeVector();
  auto gamma_shape = gamma_shape_ptr->GetShapeVector();
  auto beta_shape = beta_shape_ptr->GetShapeVector();
  const size_t x_rank = x_shape.size();

  if (IsDynamicRank(x_shape) || IsDynamicRank(gamma_shape) || IsDynamicRank(beta_shape)) {
    auto any_shape =
      std::make_shared<abstract::TensorShape>(std::vector<int64_t>{abstract::TensorShape::kShapeRankAny});
    std::vector<BaseShapePtr> shapes_list = {any_shape, any_shape, any_shape, any_shape};
    return std::make_shared<abstract::TupleShape>(shapes_list);
  }

  ShapeVector mean_var_shape = x_shape;
  mean_var_shape[x_rank - 1] = 1;
  std::vector<BaseShapePtr> shapes_list = {x_shape_ptr};
  (void)shapes_list.emplace_back(std::make_shared<abstract::TensorShape>(mean_var_shape));
  (void)shapes_list.emplace_back(std::make_shared<abstract::TensorShape>(mean_var_shape));
  (void)shapes_list.emplace_back(x_shape_ptr);

  return std::make_shared<abstract::TupleShape>(shapes_list);
}

TypePtr AddLayerNormV2FuncImpl::InferType(const PrimitivePtr &prim,
                                          const std::vector<AbstractBasePtr> &input_args) const {
  auto x_type = input_args[kInputIndex0]->GetType();
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  std::vector<TypePtr> types_list = {x_type, std::make_shared<TensorType>(kFloat32),
                                     std::make_shared<TensorType>(kFloat32), x_type};
  return std::make_shared<Tuple>(types_list);
}

ShapeArray AddLayerNormV2FuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  MS_EXCEPTION_IF_NULL(primitive);
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  const auto &x_shape = x_tensor->shape();
  const size_t x_rank = x_shape.size();
  auto mean_var_shape = x_shape;
  mean_var_shape[x_rank - 1] = 1;

  return {x_shape, mean_var_shape, mean_var_shape, x_shape};
}

TypePtrList AddLayerNormV2FuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  const auto &mean_var_type = std::make_shared<TensorType>(kFloat32);
  return {x_tensor->Dtype(), mean_var_type, mean_var_type, x_tensor->Dtype()};
}

}  // namespace ops
}  // namespace mindspore
