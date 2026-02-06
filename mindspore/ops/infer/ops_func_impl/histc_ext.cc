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

#include "infer/ops_func_impl/histc_ext.h"
#include <algorithm>
#include <memory>
#include <utility>
#include <set>
#include "abstract/dshape.h"
#include "primitive/op_name.h"
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/shape_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "primitive/auto_generate/gen_ops_primitive_h.h"

namespace mindspore {
namespace ops {
BaseShapePtr HistcExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) const {
  int64_t bins = abstract::Shape::kShapeDimAny;
  auto bins_ptr = GetScalarValue<int64_t>(input_args[kInputIndex1]->GetValue());
  if (bins_ptr.has_value()) {
    bins = bins_ptr.value();
    MS_CHECK_VALUE(bins > 0, "For 'histc', attr 'bins' value should greater than 0. but got " + std::to_string(bins));
  }
  return std::make_shared<abstract::Shape>(ShapeVector{bins});
}

TypePtr HistcExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  auto x_type = input_args[kInputIndex0]->GetType();
  return x_type->Clone();
}

ShapeArray HistcExtFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  int64_t bins = abstract::Shape::kShapeDimAny;
  auto bins_ptr = GetScalarValue<int64_t>(input_values[kInputIndex1]);
  if (bins_ptr.has_value()) {
    bins = bins_ptr.value();
    MS_CHECK_VALUE(bins > 0, "For 'histc', attr 'bins' value should greater than 0. but got " + std::to_string(bins));
  }
  return {ShapeVector{bins}};
}

TypePtrList HistcExtFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  return {x_tensor->Dtype()};
}
REGISTER_SIMPLE_INFER(kNameHistcExt, HistcExtFuncImpl)
}  // namespace ops
}  // namespace mindspore
