/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
  auto prim_name = primitive->name();

  // Infer type
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto x_type = input_args[0]->BuildType()->cast<TensorTypePtr>()->element();
  // Infer shape
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("x_shape", SizeToLong(x_shape.size()), kGreaterThan, 0, prim_name);
  auto shp = x_shape;
  auto segment_ids_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("segment_ids_shape", SizeToLong(segment_ids_shape.size()), kGreaterThan, 0,
                                           prim_name);
  CheckAndConvertUtils::Check("input_x", int64_t(x_shape.size()), kGreaterEqual, "segment_ids_shape",
                              int64_t(segment_ids_shape.size()), prim_name);

  if ((x_shape.end() != find(x_shape.begin(), x_shape.end(), -1)) &&
      (segment_ids_shape.end() != find(segment_ids_shape.begin(), segment_ids_shape.end(), -1))) {
    size_t size = segment_ids_shape.size();
    for (size_t i = 0; i < size; ++i) {
      CheckAndConvertUtils::Check("segment_ids_shp", segment_ids_shape[i], kEqual, "x_shape", x_shape[i], prim_name);
    }
  }

  const std::set<TypePtr> valid_num_segments_types = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("num_segments", input_args[kInputIndex2]->BuildType(),
                                                   valid_num_segments_types, prim_name);
  size_t size_segment_ids_shp = segment_ids_shape.size();
  size_t size_x_shape = x_shape.size();
  for (size_t i = size_segment_ids_shp; i < size_x_shape; ++i) {
    (void)shp.emplace_back(x_shape[i]);
  }

  return std::make_shared<abstract::AbstractTensor>(x_type, shp);
}
REGISTER_PRIMITIVE_C(kNameUnsortedSegmentSum, UnsortedSegmentSum);
}  // namespace ops
}  // namespace mindspore
