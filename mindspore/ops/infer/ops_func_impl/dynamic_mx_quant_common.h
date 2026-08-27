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

#ifndef MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_DYNAMIC_MX_QUANT_COMMON_H_
#define MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_DYNAMIC_MX_QUANT_COMMON_H_

#include <optional>
#include <string>
#include <vector>
#include "ops/ops_func_impl/op_func_impl.h"

namespace mindspore::ops::dynamic_mx_quant {
constexpr int64_t kDstTypeFp8E5M2 = 23;
constexpr int64_t kDstTypeFp8E5M2Alias = 291;
constexpr int64_t kDstTypeFp8E4M3Fn = 24;
constexpr int64_t kDstTypeFp8E4M3FnAlias = 292;
constexpr int64_t kMxBlockSize = 32;

void CheckInputType(TypeId input_type, const std::string &op_name);
void CheckInputRank(const ShapeVector &input_shape, size_t min_rank, const std::string &op_name);
void CheckRoundMode(const std::string &round_mode, const std::string &op_name);
void CheckDstType(int64_t dst_type, const std::string &op_name);
void CheckScaleAlg(int64_t scale_alg, const std::string &op_name);
void CheckDualAxisScaleAlg(int64_t scale_alg, const std::string &op_name);
void CheckAxis(int64_t axis, size_t rank, const std::string &op_name);
void CheckBlockSize(int64_t block_size, const std::string &op_name);

ShapeVector InferScaleShape(const ShapeVector &input_shape, const std::optional<int64_t> &axis,
                            const std::optional<int64_t> &block_size);
TypeId InferQuantizedType(const std::optional<int64_t> &dst_type, const std::string &op_name);
}  // namespace mindspore::ops::dynamic_mx_quant

#endif  // MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_DYNAMIC_MX_QUANT_COMMON_H_
