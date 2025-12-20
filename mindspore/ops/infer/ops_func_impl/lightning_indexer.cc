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

#include "infer/ops_func_impl/lightning_indexer.h"
#include <algorithm>
#include <string>
#include "op_def/op_enum.h"
#include "ops_utils/op_constants.h"
#include "ops_utils/op_utils.h"

namespace mindspore {
namespace ops {
ShapeArray LightningIndexerFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                const InferInfoPtrList &input_infos) const {
  auto &query = input_infos[kIndex0];
  auto &key = input_infos[kIndex1];
  const auto &query_shape = query->GetShape();
  const auto &key_shape = key->GetShape();

  auto is_shape_empty = [](const ShapeVector &shape) -> bool {
    return std::any_of(shape.begin(), shape.end(), [](int64_t dim) { return dim == 0; });
  };

  MS_CHECK_VALUE(!is_shape_empty(query_shape), "For LightningIndexer, the shape of 'query' cannot be empty.");
  MS_CHECK_VALUE(!is_shape_empty(key_shape), "For LightningIndexer, the shape of 'key' cannot be empty.");

  constexpr size_t kRankBSND = 4;
  constexpr size_t kRankTND = 3;
  auto is_query_dynamic_rank = query->IsDynamicRank();
  if (!is_query_dynamic_rank) {
    MS_CHECK_VALUE(
      query_shape.size() == kRankBSND || query_shape.size() == kRankTND,
      "For LightningIndexer, the rank of 'query' must be 3 or 4, but got " + std::to_string(query_shape.size()) + ".");
  }

  auto is_key_dynamic_rank = key->IsDynamicRank();
  if (!is_key_dynamic_rank) {
    MS_CHECK_VALUE(
      key_shape.size() == kRankBSND || key_shape.size() == kRankTND,
      "For LightningIndexer, the rank of 'key' must be 3 or 4, but got " + std::to_string(key_shape.size()) + ".");
  }

  auto &sparse_count = input_infos[kIndex8];
  auto sparse_count_opt = sparse_count->GetScalarValue<int64_t>();
  if (sparse_count_opt.has_value()) {
    MS_CHECK_VALUE(sparse_count_opt.value() > 0,
                   "For LightningIndexer, the sparse count must be greater than 0, but got " +
                     std::to_string(sparse_count_opt.value()) + ".");
  }

  auto &layout_query = input_infos[kIndex6];
  auto layout_query_opt = layout_query->GetScalarValue<int64_t>();
  if (MS_UNLIKELY(!layout_query_opt.has_value())) {
    return {ShapeVector{abstract::TensorShape::kShapeRankAny}, ShapeVector{abstract::TensorShape::kShapeRankAny}};
  }

  ShapeVector output_shape;
  constexpr int64_t kDimUnknown = abstract::TensorShape::kShapeDimAny;
  auto layout_query_value = static_cast<FASInputLayoutMode>(layout_query_opt.value());
  if (layout_query_value == FASInputLayoutMode::BSND) {
    output_shape.reserve(kRankBSND);
    output_shape.push_back(is_query_dynamic_rank ? kDimUnknown : query_shape[kIndex0]);
    output_shape.push_back(is_query_dynamic_rank ? kDimUnknown : query_shape[kIndex1]);
    output_shape.push_back(is_key_dynamic_rank ? kDimUnknown : key_shape[kIndex2]);
  } else {
    output_shape.reserve(kRankTND);
    output_shape.push_back(is_query_dynamic_rank ? kDimUnknown : query_shape[kIndex0]);
    auto &layout_key = input_infos[kIndex7];
    auto layout_key_opt = layout_key->GetScalarValue<int64_t>();
    if (MS_UNLIKELY(!layout_key_opt.has_value() || is_key_dynamic_rank)) {
      output_shape.push_back(kDimUnknown);
    } else {
      auto layout_key_value = static_cast<FASInputLayoutMode>(layout_key_opt.value());
      output_shape.push_back(layout_key_value == FASInputLayoutMode::TND ? key_shape[kIndex1] : key_shape[kIndex2]);
    }
  }
  output_shape.push_back(!sparse_count_opt.has_value() ? kDimUnknown : sparse_count_opt.value());
  return {output_shape, output_shape};
}

std::vector<TypeId> LightningIndexerFuncImpl::InferType(const PrimitivePtr &primitive,
                                                        const InferInfoPtrList &input_infos) const {
  auto query_type = input_infos[kIndex0]->GetType();
  return {kNumberTypeInt32, query_type};
}
}  // namespace ops
}  // namespace mindspore
