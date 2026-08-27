/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/dynamic_mx_quant_with_dual_axis.h"
#include "infer/ops_func_impl/dynamic_mx_quant_common.h"
#include "ops_utils/op_constants.h"

namespace mindspore::ops {
namespace {
constexpr size_t kDynamicMxQuantDualAxisMinRank = 2;
constexpr size_t kDynamicMxQuantDualAxisDstTypeIndex = 1;
constexpr size_t kDynamicMxQuantDualAxisRoundModeIndex = 2;
constexpr size_t kDynamicMxQuantDualAxisScaleAlgIndex = 3;
}  // namespace

int32_t DynamicMxQuantWithDualAxisFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                                            const InferInfoPtrList &input_infos) const {
  const auto &input = input_infos[kIndex0];
  bool need_retry = false;
  const auto round_mode = input_infos[kDynamicMxQuantDualAxisRoundModeIndex]->GetScalarValue<std::string>();
  if (round_mode.has_value()) {
    dynamic_mx_quant::CheckRoundMode(round_mode.value(), primitive->name());
  } else {
    need_retry = true;
  }

  const auto scale_alg = input_infos[kDynamicMxQuantDualAxisScaleAlgIndex]->GetScalarValue<int64_t>();
  if (scale_alg.has_value()) {
    dynamic_mx_quant::CheckDualAxisScaleAlg(scale_alg.value(), primitive->name());
  } else {
    need_retry = true;
  }

  const auto dst_type = input_infos[kDynamicMxQuantDualAxisDstTypeIndex]->GetScalarValue<int64_t>();
  if (dst_type.has_value()) {
    dynamic_mx_quant::CheckDstType(dst_type.value(), primitive->name());
  } else {
    need_retry = true;
  }

  if (input->IsDynamicRank()) {
    return OP_CHECK_RETRY;
  }

  const auto &input_shape = input->GetShape();
  dynamic_mx_quant::CheckInputRank(input_shape, kDynamicMxQuantDualAxisMinRank, primitive->name());
  return need_retry ? OP_CHECK_RETRY : OP_CHECK_SUCCESS;
}

ShapeArray DynamicMxQuantWithDualAxisFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                          const InferInfoPtrList &input_infos) const {
  const auto &input = input_infos[kIndex0];
  if (input->IsDynamicRank()) {
    const ShapeVector dynamic_rank = {abstract::Shape::kShapeRankAny};
    return {dynamic_rank, dynamic_rank, dynamic_rank, dynamic_rank};
  }

  const auto &input_shape = input->GetShape();
  dynamic_mx_quant::CheckInputRank(input_shape, kDynamicMxQuantDualAxisMinRank, primitive->name());
  const auto dst_type = input_infos[kDynamicMxQuantDualAxisDstTypeIndex]->GetScalarValue<int64_t>();
  if (dst_type.has_value()) {
    dynamic_mx_quant::CheckDstType(dst_type.value(), primitive->name());
  }
  const auto scale1_shape = dynamic_mx_quant::InferScaleShape(input_shape, std::optional<int64_t>{-1},
                                                              std::optional<int64_t>{dynamic_mx_quant::kMxBlockSize});
  const auto scale2_shape = dynamic_mx_quant::InferScaleShape(input_shape, std::optional<int64_t>{-2},
                                                              std::optional<int64_t>{dynamic_mx_quant::kMxBlockSize});
  return {input_shape, scale1_shape, input_shape, scale2_shape};
}

std::vector<TypeId> DynamicMxQuantWithDualAxisFuncImpl::InferType(const PrimitivePtr &primitive,
                                                                  const InferInfoPtrList &input_infos) const {
  dynamic_mx_quant::CheckInputType(input_infos[kIndex0]->GetType(), primitive->name());
  const auto dst_type = input_infos[kDynamicMxQuantDualAxisDstTypeIndex]->GetScalarValue<int64_t>();
  const auto y_type = dynamic_mx_quant::InferQuantizedType(dst_type, primitive->name());
  return {y_type, kNumberTypeUInt8, y_type, kNumberTypeUInt8};
}
}  // namespace mindspore::ops
