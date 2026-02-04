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
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "infer/ops_func_impl/mse_loss_grad_ext.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"

namespace mindspore {
namespace ops {

BaseShapePtr MSELossGradExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) const {
  const auto &input_target_broadcast_shape =
    CalBroadCastShape(input_args[kInputIndex1]->GetShape()->GetShapeVector(),
                      input_args[kInputIndex2]->GetShape()->GetShapeVector(), primitive->name(), "input", "target");

  return std::make_shared<abstract::Shape>(input_target_broadcast_shape);
}

TypePtr MSELossGradExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) const {
  const auto &dout_type = input_args[kInputIndex0]->GetType();
  const auto &out_type = input_args[kInputIndex1]->GetType();
  const auto &target_type = input_args[kInputIndex2]->GetType();

  MS_EXCEPTION_IF_CHECK_FAIL(dout_type->type_id() == out_type->type_id(), "dout input type not equal");
  MS_EXCEPTION_IF_CHECK_FAIL(out_type->type_id() == target_type->type_id(), "input target type not equal");

  return dout_type;
}

ShapeArray MSELossGradExtFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &input_tensor = input_values[kIndex1]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  const auto &target_tensor = input_values[kIndex2]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(target_tensor);
  const auto &broadcast_shape =
    CalBroadCastShapeV2(input_tensor->shape(), target_tensor->shape(), primitive->name(), "input", "target");

  return {broadcast_shape};
}

TypePtrList MSELossGradExtFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &input_tensor = input_values[kIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  return {input_tensor->Dtype()};
}

REGISTER_SIMPLE_INFER(kNameMSELossGradExt, MSELossGradExtFuncImpl)

}  // namespace ops
}  // namespace mindspore
