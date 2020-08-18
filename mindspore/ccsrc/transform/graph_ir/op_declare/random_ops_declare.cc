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

#include "transform/graph_ir/op_declare/random_ops_declare.h"

namespace mindspore::transform {
// DropOutGenMask
INPUT_MAP(DropOutGenMask) = {{1, INPUT_DESC(shape)}, {2, INPUT_DESC(prob)}};
ATTR_MAP(DropOutGenMask) = {{"Seed0", ATTR_DESC(seed, AnyTraits<int64_t>())},
                            {"Seed1", ATTR_DESC(seed2, AnyTraits<int64_t>())}};
OUTPUT_MAP(DropOutGenMask) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(DropOutGenMask, prim::kPrimDropoutGenMask->name(), ADPT_DESC(DropOutGenMask))

// RandomChoiceWithMask
INPUT_MAP(RandomChoiceWithMask) = {{1, INPUT_DESC(x)}};
ATTR_MAP(RandomChoiceWithMask) = {{"count", ATTR_DESC(count, AnyTraits<int64_t>())},
                                  {"seed", ATTR_DESC(seed, AnyTraits<int64_t>())},
                                  {"seed2", ATTR_DESC(seed2, AnyTraits<int64_t>())}};
OUTPUT_MAP(RandomChoiceWithMask) = {{0, OUTPUT_DESC(y)}, {1, OUTPUT_DESC(mask)}};
REG_ADPT_DESC(RandomChoiceWithMask, kNameRandomChoiceWithMask, ADPT_DESC(RandomChoiceWithMask))

// TruncatedNormal
INPUT_MAP(TruncatedNormal) = {{1, INPUT_DESC(shape)}};
ATTR_MAP(TruncatedNormal) = {{"seed", ATTR_DESC(seed, AnyTraits<int64_t>())},
                             {"seed2", ATTR_DESC(seed2, AnyTraits<int64_t>())}};
OUTPUT_MAP(TruncatedNormal) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(TruncatedNormal, kNameTruncatedNormal, ADPT_DESC(TruncatedNormal))
}  // namespace mindspore::transform
