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

void SelectInferShapeCheck(const std::vector<int64_t> &x_shape, const std::vector<int64_t> &y_shape,
                           const std::vector<int64_t> &cond_shape, size_t shape_size) {
  for (size_t i = 0; i < shape_size; i++) {
    if ((x_shape[i] > 0 && cond_shape[i] > 0 && x_shape[i] != cond_shape[i]) ||
        (x_shape[i] > 0 && y_shape[i] > 0 && x_shape[i] != y_shape[i])) {
      MS_EXCEPTION(ValueError)
        << "For 'Select', the shape of 'condition', 'x' and 'y' must be the same. But got 'condition' shape: "
        << cond_shape << ", 'x' shape: " << x_shape << ", 'y' shape: " << y_shape << ".";
    }
  }
}

abstract::BaseShapePtr SelectFuncImpl::InferShape(const PrimitivePtr &prim,
                                                  const std::vector<AbstractBasePtr> &input_args) const {
  auto cond_shape = input_args[kSelectCondIndex]->GetShape()->GetShapeVector();
  auto x_shape = input_args[kSelectXIndex]->GetShape()->GetShapeVector();
  auto y_shape = input_args[kSelectYIndex]->GetShape()->GetShapeVector();
  if (IsDynamicRank(cond_shape) || IsDynamicRank(x_shape) || IsDynamicRank(y_shape)) {
    return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::TensorShape::kShapeRankAny});
  }
  auto cond_shape_size = cond_shape.size();
  auto x_shape_size = x_shape.size();
  auto y_shape_size = y_shape.size();
  if (cond_shape_size != x_shape_size || y_shape_size != x_shape_size) {
    MS_EXCEPTION(ValueError)
      << "For 'Select', the shape of 'condition', 'x' and 'y' must be the same. But got 'condition' shape: "
      << cond_shape << ", 'x' shape: " << x_shape << ", 'y' shape: " << y_shape << ".";
  }
  SelectInferShapeCheck(x_shape, y_shape, cond_shape, x_shape_size);
  return input_args[kSelectCondIndex]->GetShape()->Clone();
}

TypePtr SelectFuncImpl::InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = prim->name();
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
  if (*x_type != *y_type) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name
                            << "', the x_type and y_type must be the same, but got x_type: " << x_type->ToString()
                            << " and y_type: " << y_type->ToString() << ".";
  }
  return x_type->Clone();
}
}  // namespace ops
}  // namespace mindspore
