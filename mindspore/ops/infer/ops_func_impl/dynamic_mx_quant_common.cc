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

#include "infer/ops_func_impl/dynamic_mx_quant_common.h"
#include <algorithm>
#include <set>
#include "ops_utils/op_constants.h"
#include "ops_utils/op_utils.h"

namespace mindspore::ops::dynamic_mx_quant {
namespace {
constexpr int64_t kMinScaleAlg = 0;
constexpr int64_t kMaxScaleAlg = 2;

int64_t InferScaleDim(int64_t dim, int64_t block_size) {
  if (dim == abstract::Shape::kShapeDimAny) {
    return abstract::Shape::kShapeDimAny;
  }
  const auto block_count = (dim + block_size - 1) / block_size;
  return (block_count + 1) / 2;
}
}  // namespace

void CheckInputType(TypeId input_type, const std::string &op_name) {
  if (input_type != kNumberTypeFloat16 && input_type != kNumberTypeBFloat16) {
    MS_EXCEPTION(TypeError) << "For " << op_name << ", 'input' must have dtype Float16 or BFloat16, but got "
                            << TypeIdToString(input_type) << ".";
  }
}

void CheckInputRank(const ShapeVector &input_shape, size_t min_rank, const std::string &op_name) {
  constexpr size_t kMaxRank = 7;
  MS_CHECK_VALUE(input_shape.size() >= min_rank && input_shape.size() <= kMaxRank,
                 "For " + op_name + ", the rank of 'input' must be in [" + std::to_string(min_rank) + ", " +
                   std::to_string(kMaxRank) + "], but got " + std::to_string(input_shape.size()) + ".");
}

void CheckRoundMode(const std::string &round_mode, const std::string &op_name) {
  MS_CHECK_VALUE(
    round_mode == "rint",
    "For " + op_name + ", the runtime-tested value of 'round_mode' is 'rint', but got '" + round_mode + "'.");
}

void CheckDstType(int64_t dst_type, const std::string &op_name) {
  static const std::set<int64_t> kValidDstTypes = {kDstTypeFp8E5M2, kDstTypeFp8E4M3Fn, kDstTypeFp8E5M2Alias,
                                                   kDstTypeFp8E4M3FnAlias};
  MS_CHECK_VALUE(kValidDstTypes.count(dst_type) != 0,
                 "For " + op_name + ", the supported values of 'dst_type' are [23, 24, 291, 292], but got " +
                   std::to_string(dst_type) + ".");
}

void CheckScaleAlg(int64_t scale_alg, const std::string &op_name) {
  MS_CHECK_VALUE(scale_alg >= kMinScaleAlg && scale_alg <= kMaxScaleAlg,
                 "For " + op_name + ", 'scale_alg' must be in [0, 2], but got " + std::to_string(scale_alg) + ".");
}

void CheckDualAxisScaleAlg(int64_t scale_alg, const std::string &op_name) {
  constexpr int64_t kMaxDualAxisScaleAlg = 1;
  MS_CHECK_VALUE(scale_alg >= kMinScaleAlg && scale_alg <= kMaxDualAxisScaleAlg,
                 "For " + op_name + ", the runtime-tested values of 'scale_alg' are [0, 1], but got " +
                   std::to_string(scale_alg) + ".");
}

void CheckAxis(int64_t axis, size_t rank, const std::string &op_name) {
  const auto rank_value = SizeToLong(rank);
  MS_CHECK_VALUE(axis >= -rank_value && axis < rank_value,
                 "For " + op_name + ", 'axis' must be in [" + std::to_string(-rank_value) + ", " +
                   std::to_string(rank_value) + "), but got " + std::to_string(axis) + ".");
}

void CheckBlockSize(int64_t block_size, const std::string &op_name) {
  MS_CHECK_VALUE(
    block_size == kMxBlockSize,
    "For " + op_name + ", the runtime-tested value of 'block_size' is 32, but got " + std::to_string(block_size) + ".");
}

ShapeVector InferScaleShape(const ShapeVector &input_shape, const std::optional<int64_t> &axis,
                            const std::optional<int64_t> &block_size) {
  ShapeVector scale_shape = input_shape;
  if (!axis.has_value()) {
    std::fill(scale_shape.begin(), scale_shape.end(), abstract::Shape::kShapeDimAny);
  } else {
    const auto normalized_axis = axis.value() < 0 ? axis.value() + SizeToLong(input_shape.size()) : axis.value();
    scale_shape[LongToSize(normalized_axis)] =
      block_size.has_value() ? InferScaleDim(scale_shape[LongToSize(normalized_axis)], block_size.value())
                             : abstract::Shape::kShapeDimAny;
  }
  scale_shape.push_back(2);
  return scale_shape;
}

TypeId InferQuantizedType(const std::optional<int64_t> &dst_type, const std::string &op_name) {
  if (!dst_type.has_value()) {
    return kTypeUnknown;
  }
  const auto dst_type_value = dst_type.value();
  CheckDstType(dst_type_value, op_name);
  if (dst_type_value == kDstTypeFp8E5M2 || dst_type_value == kDstTypeFp8E5M2Alias) {
    return kNumberTypeFloat8E5M2;
  }
  return kNumberTypeFloat8E4M3FN;
}
}  // namespace mindspore::ops::dynamic_mx_quant
