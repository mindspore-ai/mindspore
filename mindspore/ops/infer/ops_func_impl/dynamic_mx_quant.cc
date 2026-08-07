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

#include "infer/ops_func_impl/dynamic_mx_quant.h"
#include "infer/ops_func_impl/dynamic_mx_quant_common.h"
#include "ops_utils/op_constants.h"

namespace mindspore::ops {
namespace {
constexpr size_t kDynamicMxQuantMinRank = 1;
constexpr size_t kDynamicMxQuantDstTypeIndex = 1;
constexpr size_t kAxisIndex = 2;
constexpr size_t kRoundModeIndex = 3;
constexpr size_t kBlockSizeIndex = 4;
constexpr size_t kDynamicMxQuantScaleAlgIndex = 5;
}  // namespace

int32_t DynamicMxQuantFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                                const InferInfoPtrList &input_infos) const {
  const auto &input = input_infos[kIndex0];
  bool need_retry = false;
  const auto round_mode = input_infos[kRoundModeIndex]->GetScalarValue<std::string>();
  if (round_mode.has_value()) {
    dynamic_mx_quant::CheckRoundMode(round_mode.value(), primitive->name());
  } else {
    need_retry = true;
  }

  const auto block_size = input_infos[kBlockSizeIndex]->GetScalarValue<int64_t>();
  if (block_size.has_value()) {
    dynamic_mx_quant::CheckBlockSize(block_size.value(), primitive->name());
  } else {
    need_retry = true;
  }

  const auto scale_alg = input_infos[kDynamicMxQuantScaleAlgIndex]->GetScalarValue<int64_t>();
  if (scale_alg.has_value()) {
    dynamic_mx_quant::CheckScaleAlg(scale_alg.value(), primitive->name());
  } else {
    need_retry = true;
  }

  const auto dst_type = input_infos[kDynamicMxQuantDstTypeIndex]->GetScalarValue<int64_t>();
  if (dst_type.has_value()) {
    dynamic_mx_quant::CheckDstType(dst_type.value(), primitive->name());
  } else {
    need_retry = true;
  }

  if (input->IsDynamicRank()) {
    return OP_CHECK_RETRY;
  }

  const auto &input_shape = input->GetShape();
  dynamic_mx_quant::CheckInputRank(input_shape, kDynamicMxQuantMinRank, primitive->name());
  const auto axis = input_infos[kAxisIndex]->GetScalarValue<int64_t>();
  if (axis.has_value()) {
    dynamic_mx_quant::CheckAxis(axis.value(), input_shape.size(), primitive->name());
  } else {
    need_retry = true;
  }
  return need_retry ? OP_CHECK_RETRY : OP_CHECK_SUCCESS;
}

ShapeArray DynamicMxQuantFuncImpl::InferShape(const PrimitivePtr &primitive,
                                              const InferInfoPtrList &input_infos) const {
  const auto &input = input_infos[kIndex0];
  if (input->IsDynamicRank()) {
    return {{abstract::Shape::kShapeRankAny}, {abstract::Shape::kShapeRankAny}};
  }

  const auto &input_shape = input->GetShape();
  dynamic_mx_quant::CheckInputRank(input_shape, kDynamicMxQuantMinRank, primitive->name());
  const auto axis = input_infos[kAxisIndex]->GetScalarValue<int64_t>();
  const auto dst_type = input_infos[kDynamicMxQuantDstTypeIndex]->GetScalarValue<int64_t>();
  const auto block_size = input_infos[kBlockSizeIndex]->GetScalarValue<int64_t>();
  if (axis.has_value()) {
    dynamic_mx_quant::CheckAxis(axis.value(), input_shape.size(), primitive->name());
  }
  if (block_size.has_value()) {
    dynamic_mx_quant::CheckBlockSize(block_size.value(), primitive->name());
  }
  if (dst_type.has_value()) {
    dynamic_mx_quant::CheckDstType(dst_type.value(), primitive->name());
  }
  return {input_shape, dynamic_mx_quant::InferScaleShape(input_shape, axis, block_size)};
}

std::vector<TypeId> DynamicMxQuantFuncImpl::InferType(const PrimitivePtr &primitive,
                                                      const InferInfoPtrList &input_infos) const {
  dynamic_mx_quant::CheckInputType(input_infos[kIndex0]->GetType(), primitive->name());
  const auto dst_type = input_infos[kDynamicMxQuantDstTypeIndex]->GetScalarValue<int64_t>();
  return {dynamic_mx_quant::InferQuantizedType(dst_type, primitive->name()), kNumberTypeUInt8};
}
}  // namespace mindspore::ops
