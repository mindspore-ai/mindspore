/**
 * Copyright 2022-2022 Huawei Technologies Co., Ltd
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

#include "transform/graph_ir/op_declare/cluster_ops_declare.h"
#include "mindspore/core/ops/other_ops.h"

namespace mindspore::transform {
// KMeansCentroids
INPUT_MAP(KMeansCentroids) = {
  {1, INPUT_DESC(x)}, {2, INPUT_DESC(y)}, {3, INPUT_DESC(sum_square_y)}, {4, INPUT_DESC(sum_square_x)}};
ATTR_MAP(KMeansCentroids) = {
  {"use_actual_distance", ATTR_DESC(use_actual_distance, AnyTraits<bool>(), AnyTraits<bool>())}};
OUTPUT_MAP(KMeansCentroids) = {
  {0, OUTPUT_DESC(segment_sum)}, {1, OUTPUT_DESC(segment_count)}, {2, OUTPUT_DESC(kmean_total_sum)}};
REG_ADPT_DESC(KMeansCentroids, prim::kPrimKMeansCentroids->name(), ADPT_DESC(KMeansCentroids))
}  // namespace mindspore::transform
