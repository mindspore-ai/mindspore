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

#include "infer/ops_func_impl/add_rms_norm.h"

#include <string>
#include <map>
#include "abstract/dshape.h"
#include "include/op_def.h"
#include "primitive/op_name.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
int32_t AddRmsNormFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) const {
  auto epsilon_value = GetScalarValue<pyfloat>(input_args[kInputIndex3]->GetValue());
  if (MS_UNLIKELY(!epsilon_value.has_value())) {
    return OP_CHECK_RETRY;
  }
  MS_CHECK_VALUE(epsilon_value.value() > 0 && epsilon_value.value() <= 1,
                 CheckAndConvertUtils::FormatCheckInRangeMsg<pyfloat>("epsilon", epsilon_value.value(), kIncludeRight,
                                                                      {0., 1.}, primitive));
  return OP_CHECK_SUCCESS;
}

ShapeArray AddRmsNormFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto &x1 = input_infos[kInputIndex0];
  auto &x2 = input_infos[kInputIndex1];
  auto &gamma = input_infos[kInputIndex2];
  const auto &x1_shape = x1->GetShape();
  const auto &x2_shape = x2->GetShape();
  const auto &gamma_shape = gamma->GetShape();
  auto gamma_rank = gamma_shape.size();
  MS_EXCEPTION_IF_CHECK_FAIL(!IsShapeNone(x1_shape) && !IsShapeNone(x2_shape) && !IsShapeNone(gamma_shape),
                             "For AddRmsNorm, [x1] or [x2] or [gamma] is none tensor, which is not allowed.");
  if (x1->IsDynamicRank() && x2->IsDynamicRank() && gamma->IsDynamicRank()) {
    auto out_shape = ShapeVector{abstract::Shape::kShapeRankAny};
    return {out_shape, out_shape, out_shape};
  }
  if (!(x1->IsDynamic() || x2->IsDynamic())) {
    if (x1_shape != x2_shape) {
      MS_EXCEPTION(ValueError) << "For AddRmsNorm, shape of x1: " << x1_shape
                               << " are not consistent with the shape x2: " << x2_shape << " .";
    }
  }
  auto out_shape = InferOutShapeSameAsInShape({x1_shape, x2_shape});
  auto out_rank = out_shape.size();
  auto rstd_shape = out_shape;
  if (gamma->IsDynamicRank()) {
    if (!IsDynamicRank(out_shape)) {
      rstd_shape = ShapeVector(out_rank, abstract::TensorShape::kShapeDimAny);
    } else {
      rstd_shape = ShapeVector{abstract::TensorShape::kShapeRankAny};
    }
  } else if (!IsDynamicRank(out_shape)) {
    MS_EXCEPTION_IF_CHECK_FAIL(gamma_rank <= out_rank,
                               "For AddRmsNorm, The [gamma] rank can not be bigger than the rank of other two inputs."
                               "But got: " +
                                 std::to_string(gamma_rank) + " vs " + std::to_string(out_rank));
    for (auto dim = out_rank - gamma_rank; dim < out_rank; dim++) {
      int64_t x_dim = out_shape[dim];
      int64_t gamma_dim = gamma_shape[dim - out_rank + gamma_rank];
      MS_EXCEPTION_IF_CHECK_FAIL(
        x_dim == gamma_dim || x_dim == abstract::TensorShape::kShapeDimAny ||
          gamma_dim == abstract::TensorShape::kShapeDimAny,
        "For AddRmsNorm, Each dimension of [gamma] must be aligned to the corresponding dimension of other two inputs. "
        "But got: " +
          std::to_string(gamma_dim) + " vs " + std::to_string(x_dim));
      rstd_shape[dim] = 1;
      if (x_dim == abstract::TensorShape::kShapeDimAny && gamma_dim != abstract::TensorShape::kShapeDimAny) {
        out_shape[dim] = gamma_dim;
      }
    }
  }
  return {out_shape, rstd_shape, out_shape};
}

std::vector<TypeId> AddRmsNormFuncImpl::InferType(const PrimitivePtr &primitive,
                                                  const InferInfoPtrList &input_infos) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto x1_dtype_id = input_infos[kInputIndex0]->GetType();
  return {x1_dtype_id, kNumberTypeFloat32, x1_dtype_id};
}
}  // namespace ops
}  // namespace mindspore
