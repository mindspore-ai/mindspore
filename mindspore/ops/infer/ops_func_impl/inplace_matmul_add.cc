/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/inplace_matmul_add.h"

#include <algorithm>
#include <string>
#include <numeric>
#include <vector>
#include <set>

#include "ops_utils/op_constants.h"
#include "ops/op_def_utils.h"
#include "utils/check_convert_utils.h"
#include "ir/dtype.h"

namespace mindspore {
namespace ops {
namespace {
void InplaceMatmulAddCheckInputShape(const PrimitivePtr &primitive, const InferInfoPtr &arg_info,
                                     const std::vector<int64_t> &arg_shape, const std::string &arg_name) {
  if (MS_UNLIKELY(arg_info->IsDynamicRank())) {
    return;
  }
  const auto arg_rank = arg_shape.size();
  MS_CHECK_VALUE(arg_rank >= kIndex2 && arg_rank <= kIndex3,
                 CheckAndConvertUtils::FormatCheckInRangeMsg<size_t>(arg_name + "'s rank", arg_rank, kIncludeBoth,
                                                                     {kIndex2, kIndex3}, primitive));
  if (MS_UNLIKELY(arg_info->IsDynamic())) {
    return;
  }
  if (std::any_of(arg_shape.begin(), arg_shape.end(), [](int64_t dim_size) { return dim_size <= 0; })) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << ", the input `" << arg_name
                             << "` should not be empty, but got " << arg_name << "'s shape: " << arg_shape;
  }
}

std::vector<int64_t> InplaceMatmulAddInferShape(const PrimitivePtr &primitive, const InferInfoPtr &x_info,
                                                const std::vector<int64_t> &x_shape, const InferInfoPtr &weight_info,
                                                const std::vector<int64_t> &weight_shape) {
  if (MS_UNLIKELY(x_info->IsDynamic() || weight_info->IsDynamic())) {
    return std::vector<int64_t>{abstract::Shape::kShapeRankAny};
  }

  const auto &prim_name = primitive->name();
  const auto x_rank = x_shape.size();
  const auto &weight_rank = weight_shape.size();
  if (MS_UNLIKELY(x_rank != weight_rank)) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", the ranks of `x` and `weight` should be the same, but got "
                             << x_rank << " and " << weight_rank;
  }
  if (MS_UNLIKELY(x_rank == kIndex3 && x_shape.front() != weight_shape.front())) {
    MS_EXCEPTION(ValueError) << "For " << prim_name
                             << ", the batch dim of `x` and `weight` should be the same, but got " << x_shape.front()
                             << " and " << weight_shape.front();
  }

  auto k_in_x = x_shape[x_rank - kIndex2];
  auto k_in_weight = weight_shape[weight_rank - kIndex2];
  if (MS_UNLIKELY(k_in_x != k_in_weight)) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", the k dim of `x` and `weight` should be the same, but got "
                             << k_in_x << " and " << k_in_weight;
  }

  auto m = x_shape.back();
  auto n = weight_shape.back();
  std::vector<int64_t> out_shape{m, n};
  if (x_rank == kIndex3) {
    (void)out_shape.insert(out_shape.begin(), x_shape.front());
  }
  return out_shape;
}

void InplaceMatmulAddCheckShapeIsMatch(const PrimitivePtr &primitive, const InferInfoPtr &c_info,
                                       const std::vector<int64_t> &c_shape, const std::vector<int64_t> &x_shape,
                                       const std::vector<int64_t> &weight_shape,
                                       const std::vector<int64_t> &expect_c_shape) {
  if (MS_UNLIKELY(c_info->IsDynamic() || IsDynamic(expect_c_shape))) {
    return;
  }
  if (MS_UNLIKELY(expect_c_shape != c_shape)) {
    MS_EXCEPTION(ValueError) << "For " << primitive->name() << ", given x.shape: " << x_shape
                             << " and weight.shape: " << weight_shape << ", C.shape should be " << expect_c_shape
                             << ", but got " << c_shape;
  }
}
}  // namespace
void InplaceMatmulAddFuncImpl::CheckShapes(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const auto &x_info = input_infos[kIndex0];
  const auto &x_shape = x_info->GetShape();
  InplaceMatmulAddCheckInputShape(primitive, x_info, x_shape, "x");

  const auto &weight_info = input_infos[kIndex1];
  const auto &weight_shape = weight_info->GetShape();
  InplaceMatmulAddCheckInputShape(primitive, weight_info, weight_shape, "weight");

  const auto &c_info = input_infos[kIndex2];
  const auto &c_shape = c_info->GetShape();
  InplaceMatmulAddCheckInputShape(primitive, c_info, c_shape, "C");

  const auto &expect_c_shape = InplaceMatmulAddInferShape(primitive, x_info, x_shape, weight_info, weight_shape);
  InplaceMatmulAddCheckShapeIsMatch(primitive, c_info, c_shape, x_shape, weight_shape, expect_c_shape);
}

ShapeArray InplaceMatmulAddFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                const InferInfoPtrList &input_infos) const {
  CheckShapes(primitive, input_infos);
  return {input_infos[kIndex2]->GetShape()};
}

std::vector<TypeId> InplaceMatmulAddFuncImpl::InferType(const PrimitivePtr &primitive,
                                                        const InferInfoPtrList &input_infos) const {
  const std::set<TypeId> valid_types{kNumberTypeBFloat16, kNumberTypeFloat16};
  const auto &prim_name = primitive->name();

  const auto &x_type = input_infos[kIndex0]->GetType();
  const auto &weight_type = input_infos[kIndex1]->GetType();
  (void)CheckAndConvertUtils::CheckTypeIdValid("x", x_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTypeIdValid("weight", weight_type, valid_types, prim_name);
  if (x_type != weight_type) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name << "', the dtype of `x` should be the same as `weight`, but got "
                            << TypeIdToString(x_type) << " and " << TypeIdToString(weight_type);
  }

  const auto &c_type = input_infos[kIndex2]->GetType();
  (void)CheckAndConvertUtils::CheckTypeIdValid("C", c_type, {kNumberTypeFloat32}, prim_name);

  return {c_type};
}
}  // namespace ops
}  // namespace mindspore
