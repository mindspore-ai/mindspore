/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "ops/unsorted_segment_sum.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
AbstractBasePtr UnsortedSegmentSumInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto unsortedsegmentsum_prim = primitive->cast<PrimUnsortedSegmentSumPtr>();
  MS_EXCEPTION_IF_NULL(unsortedsegmentsum_prim);
  auto prim_name = unsortedsegmentsum_prim->name();

  // Infer type
  auto x_type = input_args[0]->BuildType()->cast<TensorTypePtr>()->element();
  auto num_segments_type = input_args[2]->BuildType();
  auto num_segments_v = 4;
  std::set<TypePtr> valid_x_type = {TypeIdToType(kObjectTypeTensorType)};
  CheckAndConvertUtils::CheckSubClass("input_x", input_args[0]->BuildType(), valid_x_type, prim_name);
  std::set<TypePtr> valid_segment_ids_type = {TypeIdToType(kObjectTypeTensorType)};
  CheckAndConvertUtils::CheckSubClass("segment_ids", input_args[1]->BuildType(), valid_segment_ids_type, prim_name);

  // Infer shape
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShape("x_shape", input_args[0]->BuildShape(), prim_name);
  CheckAndConvertUtils::CheckInteger("x_shape", x_shape.size(), kGreaterThan, 0, prim_name);
  auto shp = x_shape;
  auto segment_ids_shape =
    CheckAndConvertUtils::ConvertShapePtrToShape("x_shape", input_args[1]->BuildShape(), prim_name);
  CheckAndConvertUtils::CheckInteger("segment_ids_shape", segment_ids_shape.size(), kGreaterThan, 0, prim_name);
  CheckAndConvertUtils::Check("input_x", x_shape.size(), kGreaterEqual, "segment_ids_shape", segment_ids_shape.size(),
                              prim_name);

  if ((x_shape.end() != find(x_shape.begin(), x_shape.end(), -1)) &&
      (segment_ids_shape.end() != find(segment_ids_shape.begin(), segment_ids_shape.end(), -1))) {
    int64_t size = segment_ids_shape.size();
    for (int64_t i = 0; i < size; ++i) {
      CheckAndConvertUtils::Check("segment_ids_shp", segment_ids_shape[i], kEqual, "x_shape", x_shape[i], prim_name);
    }
  }

  const std::set<TypePtr> valid_segments_types = {TypeIdToType(kObjectTypeTensorType)};
  for (const auto &valid_segments_type : valid_segments_types) {
    if (IsIdentidityOrSubclass(num_segments_type, valid_segments_type)) {
      const std::set<TypeId> valid_num_segments_types = {kNumberTypeInt32, kNumberTypeInt64};
      CheckAndConvertUtils::CheckTensorTypeValid("num_segments", input_args[2]->BuildType(), valid_num_segments_types,
                                                 prim_name);
      shp = {-1};
    } else {
      CheckAndConvertUtils::CheckInteger("num_segments", num_segments_v, kGreaterThan, 0, prim_name);
      shp = {num_segments_v};
    }
  }

  int64_t size_segment_ids_shp = segment_ids_shape.size();
  int64_t size_x_shpe = x_shape.size();
  for (int64_t i = size_segment_ids_shp; i < size_x_shpe; ++i) {
    shp.emplace_back(x_shape[i]);
  }

  return std::make_shared<abstract::AbstractTensor>(x_type, shp);
}
REGISTER_PRIMITIVE_C(kNameUnsortedSegmentSum, UnsortedSegmentSum);
}  // namespace ops
}  // namespace mindspore
