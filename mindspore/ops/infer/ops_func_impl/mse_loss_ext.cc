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

#include <algorithm>
#include <memory>
#include "abstract/dshape.h"
#include "mindapi/base/types.h"
#include "primitive/op_name.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "infer/ops_func_impl/mse_loss_ext.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"

namespace mindspore {
namespace ops {

BaseShapePtr MSELossExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) const {
  auto reduction_value = input_args[kInputIndex2]->GetValue();
  auto reduction_opt = GetScalarValue<int64_t>(reduction_value);
  if (MS_UNLIKELY(!reduction_opt.has_value())) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }
  auto reduce_value_enum = static_cast<Reduction>(reduction_opt.value());

  const auto &broadcast_shape = BroadCastInferShape(primitive->name(), input_args);

  if (reduce_value_enum == Reduction::NONE) {
    return broadcast_shape;
  }
  // reducion is mean or sum, all reduce.
  return std::make_shared<abstract::Shape>(ShapeVector({}));
}

TypePtr MSELossExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) const {
  const auto &input_type = input_args[kInputIndex0]->GetType();
  const auto &target_type = input_args[kInputIndex1]->GetType();

  MS_EXCEPTION_IF_CHECK_FAIL(input_type->type_id() == target_type->type_id(), "input target type not equal");

  return input_type;
}

ShapeArray MSELossExtFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &reduction_opt = input_values[kIndex2]->cast<Int64ImmPtr>();
  MS_EXCEPTION_IF_NULL(reduction_opt);
  const int64_t &reduction = reduction_opt->value();

  const auto &input_tensor = input_values[kIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  const auto &target_tensor = input_values[kIndex1]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(target_tensor);

  const auto &broadcast_shape =
    CalBroadCastShapeV2(input_tensor->shape(), target_tensor->shape(), primitive->name(), "input", "target");

  if (reduction == Reduction::NONE) {
    return {broadcast_shape};
  }

  return {ShapeVector({})};
}

TypePtrList MSELossExtFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &input_tensor = input_values[kIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  const auto &target_tensor = input_values[kIndex1]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(target_tensor);
  const auto &input_type = input_tensor->Dtype();
  const auto &target_type = target_tensor->Dtype();

  MS_EXCEPTION_IF_CHECK_FAIL(input_type->type_id() == target_type->type_id(), "input target type not equal");
  return {input_type};
}

REGISTER_SIMPLE_INFER(kNameMSELossExt, MSELossExtFuncImpl)

}  // namespace ops
}  // namespace mindspore
