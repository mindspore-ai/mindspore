/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "transform/graph_ir/op_declare/reduce_ops_declare.h"
#include <vector>

namespace mindspore::transform {
// ReduceSumD
INPUT_MAP(ReduceSumD) = {{1, INPUT_DESC(x)}};
INPUT_ATTR_MAP(ReduceSumD) = {
  {2, ATTR_DESC(axes, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};
ATTR_MAP(ReduceSumD) = {{"keep_dims", ATTR_DESC(keep_dims, AnyTraits<bool>())}};
OUTPUT_MAP(ReduceSumD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ReduceSumD, prim::kPrimReduceSum->name(), ADPT_DESC(ReduceSumD))

// ReduceProdD
INPUT_MAP(ReduceProdD) = {{1, INPUT_DESC(x)}};
INPUT_ATTR_MAP(ReduceProdD) = {
  {2, ATTR_DESC(axes, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};
ATTR_MAP(ReduceProdD) = {{"keep_dims", ATTR_DESC(keep_dims, AnyTraits<bool>())}};
OUTPUT_MAP(ReduceProdD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ReduceProdD, kNameReduceProd, ADPT_DESC(ReduceProdD))

// ReduceAllD
INPUT_MAP(ReduceAllD) = {{1, INPUT_DESC(x)}};
INPUT_ATTR_MAP(ReduceAllD) = {
  {2, ATTR_DESC(axes, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};
ATTR_MAP(ReduceAllD) = {{"keep_dims", ATTR_DESC(keep_dims, AnyTraits<bool>())}};
OUTPUT_MAP(ReduceAllD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ReduceAllD, prim::kPrimReduceAll->name(), ADPT_DESC(ReduceAllD))

// ReduceMeanD
INPUT_MAP(ReduceMeanD) = {{1, INPUT_DESC(x)}};
INPUT_ATTR_MAP(ReduceMeanD) = {
  {2, ATTR_DESC(axes, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};
ATTR_MAP(ReduceMeanD) = {{"keep_dims", ATTR_DESC(keep_dims, AnyTraits<bool>())}};
OUTPUT_MAP(ReduceMeanD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ReduceMeanD, prim::kPrimReduceMean->name(), ADPT_DESC(ReduceMeanD))

// ReduceMinD
INPUT_MAP(ReduceMinD) = {{1, INPUT_DESC(x)}};
INPUT_ATTR_MAP(ReduceMinD) = {
  {2, ATTR_DESC(axes, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};
ATTR_MAP(ReduceMinD) = {{"keep_dims", ATTR_DESC(keep_dims, AnyTraits<bool>())}};
OUTPUT_MAP(ReduceMinD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ReduceMinD, prim::kPrimReduceMin->name(), ADPT_DESC(ReduceMinD))

// ReduceMaxD
INPUT_MAP(ReduceMaxD) = {{1, INPUT_DESC(x)}};
INPUT_ATTR_MAP(ReduceMaxD) = {
  {2, ATTR_DESC(axes, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};
ATTR_MAP(ReduceMaxD) = {{"keep_dims", ATTR_DESC(keep_dims, AnyTraits<bool>())}};
OUTPUT_MAP(ReduceMaxD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ReduceMaxD, prim::kPrimReduceMax->name(), ADPT_DESC(ReduceMaxD))
}  // namespace mindspore::transform
