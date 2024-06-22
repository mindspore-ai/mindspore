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

#include <complex>
#include <memory>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "base/float16.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ir/tensor.h"
#include "mindapi/base/type_id.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/framework_ops.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "ops/ops_func_impl/select.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
using float_complex = std::complex<float>;
using double_complex = std::complex<double>;

abstract::BaseShapePtr SelectFuncImpl::InferShape(const PrimitivePtr &prim,
                                                  const std::vector<AbstractBasePtr> &input_args) const {
  if (input_args.size() < kSelectInputLen) {
    MS_LOG(EXCEPTION) << "For " << prim->name() << ", the input size should be at least" << kSelectInputLen
                      << " but got " << input_args.size();
  }
  for (size_t i = 0; i < kSelectInputLen; ++i) {
    MS_EXCEPTION_IF_NULL(input_args[i]);
    MS_EXCEPTION_IF_NULL(input_args[i]->GetShape());
  }
  auto cond_shape = input_args[kSelectCondIndex]->GetShape()->GetShapeVector();
  auto x_shape = input_args[kSelectXIndex]->GetShape()->GetShapeVector();
  auto y_shape = input_args[kSelectYIndex]->GetShape()->GetShapeVector();
  if (IsDynamicRank(cond_shape) || IsDynamicRank(x_shape) || IsDynamicRank(y_shape)) {
    return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::TensorShape::kShapeRankAny});
  }
  auto broadcast_output_size = CalBroadCastShape(x_shape, y_shape, prim->name(), "input", "other");
  auto output_size = CalBroadCastShape(cond_shape, broadcast_output_size, prim->name(), "condition", "input");
  return std::make_shared<abstract::TensorShape>(output_size);
}

ShapeArray SelectFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &cond_tensor = input_values[kSelectCondIndex]->cast<tensor::BaseTensorPtr>();
  const auto &x_tensor = input_values[kSelectXIndex]->cast<tensor::BaseTensorPtr>();
  const auto &y_tensor = input_values[kSelectYIndex]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(cond_tensor);
  MS_EXCEPTION_IF_NULL(x_tensor);
  MS_EXCEPTION_IF_NULL(y_tensor);

  const auto &cond_shape = cond_tensor->shape();
  const auto &x_shape = x_tensor->shape();
  const auto &y_shape = y_tensor->shape();
  auto broadcast_output_size = CalBroadCastShape(x_shape, y_shape, primitive->name(), "input", "other");
  auto output_size = CalBroadCastShape(cond_shape, broadcast_output_size, primitive->name(), "condition", "input");

  return {output_size};
}

TypePtr SelectFuncImpl::InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const {
  const auto &prim_name = prim->name();
  if (input_args.size() < kSelectInputLen) {
    MS_LOG(EXCEPTION) << "For " << prim->name() << ", the input size should be at least" << kSelectInputLen
                      << " but got " << input_args.size();
  }
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto x_type = input_args[kSelectXIndex]->GetType();
  auto y_type = input_args[kSelectYIndex]->GetType();
  auto cond_type = input_args[kSelectCondIndex]->GetType();
  MS_EXCEPTION_IF_NULL(x_type);
  MS_EXCEPTION_IF_NULL(y_type);

  (void)CheckAndConvertUtils::CheckTensorTypeValid("x_type", x_type, common_valid_types_with_complex_and_bool,
                                                   prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("y_type", y_type, common_valid_types_with_complex_and_bool,
                                                   prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("cond", cond_type, {kBool}, prim_name);
  return x_type->Clone();
}

TypePtrList SelectFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kSelectXIndex]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  return {x_tensor->Dtype()};
}
}  // namespace ops
}  // namespace mindspore
