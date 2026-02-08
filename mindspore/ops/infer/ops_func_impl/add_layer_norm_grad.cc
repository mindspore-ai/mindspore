/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/add_layer_norm_grad.h"
#include <complex>
#include <memory>
#include <vector>
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/helper.h"
#include "primitive/nn_ops.h"
#include "primitive/op_name.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "utils/shape_utils.h"
#include "primitive/math_ops.h"

namespace mindspore {
namespace ops {
using float_complex = std::complex<float>;
using double_complex = std::complex<double>;

abstract::BaseShapePtr AddLayerNormGradFuncImpl::InferShape(const PrimitivePtr &prim,
                                                            const std::vector<AbstractBasePtr> &input_args) const {
  const auto &dy_shape_ptr = input_args[kInputIndex0]->GetShape();
  const auto &gamma_shape_ptr = input_args[kInputIndex5]->GetShape();
  MS_EXCEPTION_IF_NULL(dy_shape_ptr);
  MS_EXCEPTION_IF_NULL(gamma_shape_ptr);
  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{dy_shape_ptr->Clone(), gamma_shape_ptr->Clone(), gamma_shape_ptr->Clone()});
}

TypePtr AddLayerNormGradFuncImpl::InferType(const PrimitivePtr &prim,
                                            const std::vector<AbstractBasePtr> &input_args) const {
  auto dx_type = input_args[kInputIndex0]->GetType();
  std::vector<TypePtr> types_list = {dx_type, std::make_shared<TensorType>(kFloat32),
                                     std::make_shared<TensorType>(kFloat32)};
  return std::make_shared<Tuple>(types_list);
}

ShapeArray AddLayerNormGradFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &dx_tensor = input_values[kInputIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(dx_tensor);
  const auto &dx_shape = dx_tensor->shape();
  const auto &gamma_tensor = input_values[kInputIndex5]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(gamma_tensor);
  const auto &gamma_shape = gamma_tensor->shape();
  return {dx_shape, gamma_shape, gamma_shape};
}

TypePtrList AddLayerNormGradFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &dx_tensor = input_values[kInputIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(dx_tensor);
  return {dx_tensor->Dtype(), std::make_shared<TensorType>(kFloat32), std::make_shared<TensorType>(kFloat32)};
}

}  // namespace ops
}  // namespace mindspore
