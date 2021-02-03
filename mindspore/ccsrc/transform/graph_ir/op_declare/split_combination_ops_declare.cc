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

#include "transform/graph_ir/op_declare/split_combination_ops_declare.h"

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
REG_ADPT_DESC(Pack, prim::kStack, ADPT_DESC(Pack))

// ConcatD
INPUT_MAP(ConcatD) = EMPTY_INPUT_MAP;
DYN_INPUT_MAP(ConcatD) = {{1, DYN_INPUT_DESC(x)}};
ATTR_MAP(ConcatD) = {
  {"axis", ATTR_DESC(concat_dim, AnyTraits<int64_t>())},
  {"inputNums", ATTR_DESC(N, AnyTraits<int64_t>())},
};
OUTPUT_MAP(ConcatD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ConcatD, prim::kPrimConcat->name(), ADPT_DESC(ConcatD))
}  // namespace mindspore::transform
