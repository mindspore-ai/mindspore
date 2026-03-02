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
#include "infer/ops_func_impl/quant_matmul.h"
#include <algorithm>
#include "ops_utils/op_constants.h"
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace ops {
int32_t QuantMatmulFuncImpl::CheckValidation(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto &scale = input_infos[kIndex2];
  auto &offset = input_infos[kIndex3];
  auto &pertoken_scale = input_infos[kIndex4];
  if (MS_UNLIKELY(scale->IsDynamicRank())) {
    return OP_CHECK_RETRY;
  }

  if (!offset->IsNone()) {
    if (MS_UNLIKELY(offset->IsDynamicRank())) {
      return OP_CHECK_RETRY;
    }
    MS_CHECK_VALUE(offset->GetShape().size() == 1, "For QuantMatmul, 'offset' must be a 1D tensor, but got shape " +
                                                     ShapeVectorToStr(offset->GetShape()));
  }

  MS_CHECK_VALUE(!pertoken_scale->IsNone(), "For QuantMatmul, 'pertoken_scale' can't be None.");

  return OP_CHECK_SUCCESS;
}

ShapeArray QuantMatmulFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  if (input_infos[kIndex0]->IsDynamicRank() || input_infos[kIndex1]->IsDynamicRank()) {
    return {ShapeVector{abstract::TensorShape::kShapeRankAny}};
  }

  auto x1_shape = input_infos[kIndex0]->GetShape();
  auto x2_shape = input_infos[kIndex1]->GetShape();
  constexpr size_t kMinRank = 2;
  MS_CHECK_VALUE(x1_shape.size() >= kMinRank && x2_shape.size() >= kMinRank,
                 "For QuantMatmul, 'x1' and 'x2' must be 2D or higher dimensional tensors, but got x1 shape: " +
                   ShapeVectorToStr(x1_shape) + ", x2 shape: " + ShapeVectorToStr(x2_shape) + ".");

  constexpr int64_t kLastToDim = 2;
  ShapeVector output_shape;
  if (x1_shape.size() > kMinRank || x2_shape.size() > kMinRank) {
    output_shape =
      CalBroadCastShape(ShapeVector(x1_shape.begin(), x1_shape.end() - kLastToDim),
                        ShapeVector(x2_shape.begin(), x2_shape.end() - kLastToDim), primitive->name(), "x1", "x2");
  }

  output_shape.push_back(x1_shape.at(SizeToLong(x1_shape.size()) - kLastToDim));
  output_shape.push_back(x2_shape.at(SizeToLong(x2_shape.size()) - 1));

  return {output_shape};
}

std::vector<TypeId> QuantMatmulFuncImpl::InferType(const PrimitivePtr &primitive,
                                                   const InferInfoPtrList &input_infos) const {
  auto x1_dtype = input_infos[kIndex0]->GetType();
  MS_CHECK_VALUE(x1_dtype != kNumberTypeInt32 && x1_dtype != kNumberTypeInt8,
                 "For QuantMatmul, the dtype of 'x1' can't be Int32 or Int8, but got " + TypeIdToString(x1_dtype));

  auto output_dtype = kNumberTypeInt8;
  if (!input_infos[kIndex6]->IsNone()) {
    auto output_dtype_opt = input_infos[kIndex6]->GetScalarValue<int64_t>();
    if (output_dtype_opt.has_value()) {
      output_dtype = static_cast<TypeId>(output_dtype_opt.value());
    }
  }
  return {output_dtype};
}
}  // namespace ops
}  // namespace mindspore
