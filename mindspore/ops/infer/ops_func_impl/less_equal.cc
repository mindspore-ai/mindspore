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
#include <string>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ir/primitive.h"
#include "mindapi/helper.h"
#include "primitive/comparison_ops.h"
#include "infer/ops_func_impl/less_equal.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"

namespace mindspore {
namespace ops {
BaseShapePtr LessEqualFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  auto op_name = primitive->name();
  return BroadCastInferShape(op_name, input_args);
}
TypePtr LessEqualFuncImpl::InferType(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  return std::make_shared<TensorType>(kBool);
}

ShapeArray LessEqualFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x1_tensor = input_values[kInputIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x1_tensor);
  const auto &x2_tensor = input_values[kInputIndex1]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x2_tensor);
  auto broadcast_shape = CalBroadCastShape(x1_tensor->shape(), x2_tensor->shape(), primitive->name());

  return {broadcast_shape};
}

TypePtrList LessEqualFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  return {kBool};
}

REGISTER_SIMPLE_INFER(kNameLessEqual, LessEqualFuncImpl)
}  // namespace ops
}  // namespace mindspore
