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
#include "infer/ops_func_impl/linalg_qr.h"
#include "mindspore/ops/ops_utils/op_utils.h"

#include "utils/check_convert_utils.h"
#include "include/op_enum.h"

namespace mindspore::ops {
ShapeArray LinalgQrFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto input_shape = input_infos[kIndex0]->GetShape();
  auto mode = input_infos[kIndex1]->GetScalarValue<int64_t>();
  const int64_t kDimLeastNum = 2;
  const int64_t kDimMaxNum = 6;

  if (input_infos[kIndex0]->IsDynamicRank()) {
    return {{abstract::Shape::kShapeRankAny}, {abstract::Shape::kShapeRankAny}};
  }
  MS_ASSERT_TRUE(input_shape.size() >= kDimLeastNum && input_shape.size() <= kDimMaxNum)
    << "For '" << primitive->name() << "', The dimension of the input tensor must be within the range [2, 6], but got "
    << input_shape.size() << "-D shape " << input_shape;

  std::vector<int64_t> out_q_shape(input_shape.begin(), input_shape.end());
  std::vector<int64_t> out_r_shape(input_shape.begin(), input_shape.end());
  const int64_t m_index = SizeToInt(input_shape.size()) - 2;
  const int64_t n_index = SizeToInt(input_shape.size()) - 1;

  if (mode == LinalgQrMode::R) {
    // Just calculate R(*, k, n) in the 'reduced' mode, where k is the minimum value of m and n,
    // and Q is returned as an empty tensor.
    out_q_shape = {};
    out_r_shape[m_index] = std::min(input_shape[m_index], input_shape[n_index]);
  } else if (mode == LinalgQrMode::REDUCED) {
    // A(*, m, n) -> Q(*, m, k); R(*, k, n); k = min(m, n)
    auto k = std::min(input_shape[m_index], input_shape[n_index]);
    out_q_shape[n_index] = k;
    out_r_shape[m_index] = k;
  } else if (mode == LinalgQrMode::COMPLETE) {
    // A(*, m, n) -> Q(*, m, m), R(*, m, n)
    out_q_shape[n_index] = input_shape[m_index];
  } else {
    MS_EXCEPTION(ValueError) << "For `LinalgQr`, the input `mode` is unsupported.";
  }

  return {out_q_shape, out_r_shape};
}

std::vector<TypeId> LinalgQrFuncImpl::InferType(const PrimitivePtr &primitive,
                                                const InferInfoPtrList &input_infos) const {
  // At present, to be consistent with the input types supported by the operator on which the backward depends,
  // the forward input type is intercepted here and only FLOAT32 is supported.
  static const std::vector<TypeId> supported_types = {kNumberTypeFloat32};
  auto input_type = input_infos[kInputIndex0]->GetType();
  bool is_supported = std::any_of(supported_types.begin(), supported_types.end(),
                                  [&input_type](const TypeId &type) { return input_type == type; });
  if (!is_supported) {
    MS_EXCEPTION(TypeError) << "For `LinalgQr`, the input tensor type is unsupported now. Please set type is 'FLOAT32'";
  }

  return {input_type, input_type};
}
}  // namespace mindspore::ops
