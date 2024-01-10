/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include <map>
#include <memory>
#include <set>
#include <vector>
#include <string>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/nn_ops.h"
#include "ops/ops_func_impl/layer_norm_grad_grad.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kInputNum = 10;
}  // namespace

TypePtr LayerNormGradGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual,
                                           SizeToLong(kInputNum), op_name);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[kInputIndex0]->GetType());
  (void)types.emplace("dy", input_args[kInputIndex1]->GetType());
  (void)types.emplace("gamma", input_args[kInputIndex4]->GetType());
  (void)types.emplace("d_dx", input_args[kInputIndex5]->GetType());
  (void)types.emplace("d_dg", input_args[kInputIndex6]->GetType());
  (void)types.emplace("d_db", input_args[kInputIndex7]->GetType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{
    input_args[kInputIndex0]->GetType(), input_args[kInputIndex1]->GetType(), input_args[kInputIndex4]->GetType()});
}

BaseShapePtr LayerNormGradGradFuncImpl::InferShape(const PrimitivePtr &,
                                                   const std::vector<AbstractBasePtr> &input_args) const {
  auto x_shape_ptr = input_args[kInputIndex0]->GetShape();
  auto x_shape = x_shape_ptr->GetShapeVector();

  auto d_dx_shape_ptr = input_args[kInputIndex1]->GetShape();
  auto d_dx_shape = d_dx_shape_ptr->GetShapeVector();
  auto dy_shape_ptr = input_args[kInputIndex5]->GetShape();
  auto dy_shape = dy_shape_ptr->GetShapeVector();
  if (!x_shape_ptr->IsDynamic() && !d_dx_shape_ptr->IsDynamic() && !dy_shape_ptr->IsDynamic()) {
    if (x_shape != d_dx_shape || x_shape != dy_shape) {
      MS_EXCEPTION(ValueError) << "For LayerNormGradGrad, x, dy, d_dx should have the same shape.";
    }
    auto gamma_shape_ptr = input_args[kInputIndex4]->GetShape();
    auto gamma_shape = gamma_shape_ptr->GetShapeVector();
    if (gamma_shape.size() < 1) {
      MS_EXCEPTION(ValueError) << "For LayerNormGradGrad, normalized shape to be at least 1-dimensional.";
    }
    auto d_dg_shape_ptr = input_args[kInputIndex6]->GetShape();
    auto d_dg_shape = d_dg_shape_ptr->GetShapeVector();
    auto d_db_shape_ptr = input_args[kInputIndex7]->GetShape();
    auto d_db_shape = d_db_shape_ptr->GetShapeVector();
    if (gamma_shape != d_dg_shape || d_dg_shape != d_db_shape) {
      MS_EXCEPTION(ValueError) << "For LayerNormGradGrad, gamma, d_dg, d_db should have the same shape.";
    }
    auto variance_shape_ptr = input_args[kInputIndex2]->GetShape();
    auto variance_shape = variance_shape_ptr->GetShapeVector();
    auto mean_shape_ptr = input_args[kInputIndex3]->GetShape();
    auto mean_shape = mean_shape_ptr->GetShapeVector();
    if (mean_shape != variance_shape) {
      MS_EXCEPTION(ValueError) << "For LayerNormGradGrad, variance, mean should have the same shape.";
    }
  }
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{
    input_args[kInputIndex0]->GetShape(), input_args[kInputIndex1]->GetShape(), input_args[kInputIndex4]->GetShape()});
}

}  // namespace ops
}  // namespace mindspore
