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

#include "infer/ops_func_impl/inplace_grouped_matmul_add.h"

#include <algorithm>
#include <numeric>
#include <vector>
#include <set>

#include "ops_utils/op_constants.h"
#include "ops/op_def_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
void InplaceGroupedMatmulAddCheckEmptyTensor(const std::string &prim_name, const std::string &arg_name,
                                             const std::vector<int64_t> &shape, bool is_dynamic) {
  if (is_dynamic) {
    return;
  }
  if (std::any_of(shape.begin(), shape.end(), [](int64_t dim_size) { return dim_size <= 0; })) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << ", the input " << arg_name << " should not be empty, but got "
                             << arg_name << "'s shape: " << shape;
  }
}
}  // namespace
ShapeArray InplaceGroupedMatmulAddFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                       const InferInfoPtrList &input_infos) const {
  return {input_infos[kIndex3]->GetShape()};
}

std::vector<TypeId> InplaceGroupedMatmulAddFuncImpl::InferType(const PrimitivePtr &primitive,
                                                               const InferInfoPtrList &input_infos) const {
  const auto &prim_name = primitive->name();
  const auto &x_type = input_infos[kIndex0]->GetType();
  const auto &weight_type = input_infos[kIndex1]->GetType();
  const std::set<TypeId> valid_types{kNumberTypeBFloat16, kNumberTypeFloat16};
  (void)CheckAndConvertUtils::CheckTypeIdValid("x", x_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTypeIdValid("weight", weight_type, valid_types, prim_name);
  if (x_type != weight_type) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name << "', the dtype of `x` should be the same as `weight`";
  }
  (void)CheckAndConvertUtils::CheckTypeIdValid("group_list", input_infos[kIndex2]->GetType(), {kNumberTypeInt64},
                                               prim_name);
  (void)CheckAndConvertUtils::CheckTypeIdValid("out", input_infos[kIndex3]->GetType(), {kNumberTypeFloat32}, prim_name);

  return {input_infos[kIndex3]->GetType()};
}

int32_t InplaceGroupedMatmulAddFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                                         const InferInfoPtrList &input_infos) const {
  const auto &prim_name = primitive->name();
  const auto &x_info = input_infos[kIndex0];
  const auto &weight_info = input_infos[kIndex1];
  const auto &group_list_info = input_infos[kIndex2];
  const auto &out_info = input_infos[kIndex3];

  const auto &x_shape = x_info->GetShape();
  InplaceGroupedMatmulAddCheckEmptyTensor(prim_name, "x", x_shape, x_info->IsDynamic());
  const auto &weight_shape = weight_info->GetShape();
  InplaceGroupedMatmulAddCheckEmptyTensor(prim_name, "weight", weight_shape, weight_info->IsDynamic());
  const auto &group_list_shape = group_list_info->GetShape();
  InplaceGroupedMatmulAddCheckEmptyTensor(prim_name, "group_list", group_list_shape, group_list_info->IsDynamic());
  const auto &out_shape = out_info->GetShape();
  InplaceGroupedMatmulAddCheckEmptyTensor(prim_name, "out", out_shape, out_info->IsDynamic());

  auto m_x = abstract::Shape::kShapeDimAny;
  auto k = abstract::Shape::kShapeDimAny;
  if (MS_LIKELY(!x_info->IsDynamicRank())) {
    MS_CHECK_VALUE(x_shape.size() == kIndex2,
                   CheckAndConvertUtils::FormatCheckIntegerMsg("x's rank", SizeToLong(x_shape.size()), kEqual,
                                                               SizeToLong(kIndex2), primitive));
    m_x = x_shape.front();
    k = x_shape.back();
  }

  auto m_weight = abstract::Shape::kShapeDimAny;
  auto n = abstract::Shape::kShapeDimAny;
  if (MS_LIKELY(!weight_info->IsDynamicRank())) {
    MS_CHECK_VALUE(weight_shape.size() == kIndex2,
                   CheckAndConvertUtils::FormatCheckIntegerMsg("grad's rank", SizeToLong(weight_shape.size()), kEqual,
                                                               SizeToLong(kIndex2), primitive));
    m_weight = weight_shape.front();
    n = weight_shape.back();
  }

  if (MS_UNLIKELY(m_x != abstract::Shape::kShapeDimAny && m_weight != abstract::Shape::kShapeDimAny &&
                  m_x != m_weight)) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', x's first dim should be equal to weight, but got " << m_x
                             << " and " << m_weight;
  }

  MS_CHECK_VALUE(group_list_shape.size() == kIndex1,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("group_list's rank", SizeToLong(group_list_shape.size()),
                                                             kEqual, SizeToLong(kIndex1), primitive));

  if (MS_LIKELY(!out_info->IsDynamicRank())) {
    auto out_rank = out_shape.size();
    MS_CHECK_VALUE(out_rank >= kIndex2 && out_rank <= kIndex3,
                   CheckAndConvertUtils::FormatCheckInRangeMsg<size_t>("out's rank", out_rank, kIncludeBoth,
                                                                       {kIndex2, kIndex3}, primitive));
  }

  std::vector<int64_t> expect_out_shape{k, n};
  if (MS_UNLIKELY(out_info->IsDynamic() || group_list_info->IsDynamic() || IsDynamic(expect_out_shape))) {
    return OP_CHECK_RETRY;
  }

  if (out_shape.size() == kIndex2) {
    expect_out_shape[kIndex0] = expect_out_shape[kIndex0] * group_list_shape[kIndex0];
  } else {
    expect_out_shape.insert(expect_out_shape.begin(), group_list_shape[kIndex0]);
  }
  if (MS_UNLIKELY(expect_out_shape != out_shape)) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', given k: " << k << ", n: " << n
                             << ", out's shape should be " << expect_out_shape << ", but got " << out_shape;
  }

  return OP_CHECK_SUCCESS;
}
}  // namespace ops
}  // namespace mindspore
