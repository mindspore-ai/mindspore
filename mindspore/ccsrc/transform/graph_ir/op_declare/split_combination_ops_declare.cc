/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "transform/graph_ir/op_declare/split_combination_ops_declare.h"
#include <vector>

namespace mindspore::transform {
// SplitD
INPUT_MAP(SplitD) = {{1, INPUT_DESC(x)}};
ATTR_MAP(SplitD) = {{"axis", ATTR_DESC(split_dim, AnyTraits<int64_t>())},
                    {"output_num", ATTR_DESC(num_split, AnyTraits<int64_t>())}};
DYN_OUTPUT_MAP(SplitD) = {{0, DYN_OUTPUT_DESC(y)}};
REG_ADPT_DESC(SplitD, kNameSplitD, ADPT_DESC(SplitD))

// Pack
INPUT_MAP(Pack) = EMPTY_INPUT_MAP;
DYN_INPUT_MAP(Pack) = {{1, DYN_INPUT_DESC(x)}};
ATTR_MAP(Pack) = {{"num", ATTR_DESC(N, AnyTraits<int64_t>())}, {"axis", ATTR_DESC(axis, AnyTraits<int64_t>())}};
OUTPUT_MAP(Pack) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Pack1, prim::kStack, ADPT_DESC(Pack))
REG_ADPT_DESC(Pack2, prim::kPack, ADPT_DESC(Pack))

// ParallelConcat
INPUT_MAP(ParallelConcat) = EMPTY_INPUT_MAP;
DYN_INPUT_MAP(ParallelConcat) = {{1, DYN_INPUT_DESC(values)}};
ATTR_MAP(ParallelConcat) = {
  {"shape", ATTR_DESC(shape, AnyTraits<std::vector<int64_t>>())},
  {"N", ATTR_DESC(N, AnyTraits<int64_t>())},
};
OUTPUT_MAP(ParallelConcat) = {{0, OUTPUT_DESC(output_data)}};
REG_ADPT_DESC(ParallelConcat, kNameParallelConcat, ADPT_DESC(ParallelConcat))

// ConcatD
INPUT_MAP(ConcatD) = EMPTY_INPUT_MAP;
DYN_INPUT_MAP(ConcatD) = {{1, DYN_INPUT_DESC(x)}};
ATTR_MAP(ConcatD) = {
  {"axis", ATTR_DESC(concat_dim, AnyTraits<int64_t>())},
  {"inputNums", ATTR_DESC(N, AnyTraits<int64_t>())},
};
OUTPUT_MAP(ConcatD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ConcatD, prim::kPrimConcat->name(), ADPT_DESC(ConcatD))

// ConcatV2 Inference for tf
DYN_INPUT_MAP(ConcatV2) = {{1, DYN_INPUT_DESC(x)}};
INPUT_MAP(ConcatV2) = {{2, INPUT_DESC(concat_dim)}};
ATTR_MAP(ConcatV2) = {
  {"N", ATTR_DESC(N, AnyTraits<int64_t>())},
};
OUTPUT_MAP(ConcatV2) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ConcatV2, kNameConcatV2, ADPT_DESC(ConcatV2))

// SplitV
INPUT_MAP(SplitV) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(size_splits)}, {3, INPUT_DESC(split_dim)}};
ATTR_MAP(SplitV) = {{"num_split", ATTR_DESC(num_split, AnyTraits<int64_t>())}};
ATTR_INPUT_MAP(SplitV) = {{"size_splits", 2}, {"split_dim", 3}};
DYN_OUTPUT_MAP(SplitV) = {{0, DYN_OUTPUT_DESC(y)}};
REG_ADPT_DESC(SplitV, prim::kPrimSplitV->name(), ADPT_DESC(SplitV))
}  // namespace mindspore::transform
