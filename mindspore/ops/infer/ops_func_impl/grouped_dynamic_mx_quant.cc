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

#include "infer/ops_func_impl/grouped_dynamic_mx_quant.h"

#include <set>
#include <string>

#include "infer/ops_func_impl/dynamic_mx_quant_common.h"
#include "ops_utils/op_constants.h"
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore::ops {
namespace {
constexpr size_t kGroupedDynamicMxQuantXIndex = 0;
constexpr size_t kGroupIndex = 1;
constexpr size_t kRoundModeIndex = 2;
constexpr size_t kDstTypeIndex = 3;
constexpr size_t kBlocksizeIndex = 4;
constexpr size_t kScaleAlgIndex = 5;
constexpr size_t kXRank = 2;
constexpr size_t kGroupIndexRank = 1;
constexpr int64_t kBlocksize = 32;
constexpr int64_t kScaleStorageBlock = 64;
constexpr int64_t kScalePairSize = 2;

void CheckRank(const PrimitivePtr &primitive, const InferInfoPtr &input, size_t expected_rank,
               const std::string &arg_name) {
  if (!input->IsDynamicRank() && input->GetShape().size() != expected_rank) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', '" << arg_name << "' must be " << expected_rank
                             << "D, but got shape " << input->GetShape() << ".";
  }
}

void CheckDstType(const PrimitivePtr &primitive, int64_t dst_type) {
  static const std::set<int64_t> kValidDstTypes = {dynamic_mx_quant::kDstTypeFp8E5M2,
                                                   dynamic_mx_quant::kDstTypeFp8E4M3Fn};
  if (MS_UNLIKELY(kValidDstTypes.count(dst_type) == 0)) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', 'dst_type' must be 23 or 24, but got " << dst_type
                             << ".";
  }
}

TypeId InferOutputType(const PrimitivePtr &primitive, int64_t dst_type) {
  CheckDstType(primitive, dst_type);
  if (dst_type == dynamic_mx_quant::kDstTypeFp8E5M2) {
    return kNumberTypeFloat8E5M2;
  }
  return kNumberTypeFloat8E4M3FN;
}
}  // namespace

int32_t GroupedDynamicMxQuantFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                                       const InferInfoPtrList &input_infos) const {
  bool need_retry = false;
  const auto round_mode = input_infos[kRoundModeIndex]->GetScalarValue<std::string>();
  if (round_mode.has_value()) {
    if (MS_UNLIKELY(round_mode.value() != "rint")) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', 'round_mode' must be \"rint\", but got \""
                               << round_mode.value() << "\".";
    }
  } else {
    need_retry = true;
  }

  const auto blocksize = input_infos[kBlocksizeIndex]->GetScalarValue<int64_t>();
  if (blocksize.has_value()) {
    if (MS_UNLIKELY(blocksize.value() != kBlocksize)) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', 'blocksize' must be 32, but got "
                               << blocksize.value() << ".";
    }
  } else {
    need_retry = true;
  }

  const auto scale_alg = input_infos[kScaleAlgIndex]->GetScalarValue<int64_t>();
  if (scale_alg.has_value()) {
    if (MS_UNLIKELY(scale_alg.value() < 0 || scale_alg.value() > 1)) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', 'scale_alg' must be 0 or 1, but got "
                               << scale_alg.value() << ".";
    }
  } else {
    need_retry = true;
  }

  const auto dst_type = input_infos[kDstTypeIndex]->GetScalarValue<int64_t>();
  if (dst_type.has_value()) {
    CheckDstType(primitive, dst_type.value());
  } else {
    need_retry = true;
  }

  CheckRank(primitive, input_infos[kGroupedDynamicMxQuantXIndex], kXRank, "x");
  CheckRank(primitive, input_infos[kGroupIndex], kGroupIndexRank, "group_index");
  if (input_infos[kGroupedDynamicMxQuantXIndex]->IsDynamicRank() || input_infos[kGroupIndex]->IsDynamicRank()) {
    return OP_CHECK_RETRY;
  }
  return need_retry ? OP_CHECK_RETRY : OP_CHECK_SUCCESS;
}

ShapeArray GroupedDynamicMxQuantFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                     const InferInfoPtrList &input_infos) const {
  const auto &x = input_infos[kGroupedDynamicMxQuantXIndex];
  const auto &group_index = input_infos[kGroupIndex];
  CheckRank(primitive, x, kXRank, "x");
  CheckRank(primitive, group_index, kGroupIndexRank, "group_index");
  if (x->IsDynamicRank() || group_index->IsDynamicRank()) {
    return {{abstract::Shape::kShapeRankAny}, {abstract::Shape::kShapeRankAny}};
  }

  const auto y_shape = x->GetShape();
  const auto dst_type = input_infos[kDstTypeIndex]->GetScalarValue<int64_t>();
  if (dst_type.has_value()) {
    CheckDstType(primitive, dst_type.value());
  }

  const auto m = x->GetShape()[kIndex0];
  const auto n = x->GetShape()[kIndex1];
  const auto g = group_index->GetShape()[kIndex0];
  const auto scale_first_dim = (m == abstract::Shape::kShapeDimAny || g == abstract::Shape::kShapeDimAny)
                                 ? abstract::Shape::kShapeDimAny
                                 : m / kScaleStorageBlock + g;
  return {y_shape, {scale_first_dim, n, kScalePairSize}};
}

TypeIdList GroupedDynamicMxQuantFuncImpl::InferType(const PrimitivePtr &primitive,
                                                    const InferInfoPtrList &input_infos) const {
  const auto x_type = input_infos[kGroupedDynamicMxQuantXIndex]->GetType();
  if (MS_UNLIKELY(x_type != kNumberTypeFloat16 && x_type != kNumberTypeBFloat16)) {
    MS_EXCEPTION(TypeError) << "For '" << primitive->name() << "', 'x' must have float16 or bfloat16 dtype, but got "
                            << TypeIdToString(x_type) << ".";
  }
  const auto group_index_type = input_infos[kGroupIndex]->GetType();
  if (MS_UNLIKELY(group_index_type != kNumberTypeInt32)) {
    MS_EXCEPTION(TypeError) << "For '" << primitive->name() << "', 'group_index' must have int32 dtype, but got "
                            << TypeIdToString(group_index_type) << ".";
  }
  const auto dst_type = input_infos[kDstTypeIndex]->GetScalarValue<int64_t>();
  return {dst_type.has_value() ? InferOutputType(primitive, dst_type.value()) : kTypeUnknown, kNumberTypeUInt8};
}
}  // namespace mindspore::ops
