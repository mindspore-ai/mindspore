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

#include "transform/graph_ir/op_declare/hcom_ops_declare.h"
#include <string>
#include <vector>

namespace mindspore::transform {
// HCOMAllreduce
INPUT_MAP(HcomAllReduce) = {{1, INPUT_DESC(x)}};
OUTPUT_MAP(HcomAllReduce) = {{0, OUTPUT_DESC(y)}};
ATTR_MAP(HcomAllReduce) = {{"op", ATTR_DESC(reduction, AnyTraits<std::string>())},
                           {"group", ATTR_DESC(group, AnyTraits<std::string>())}};
REG_ADPT_DESC(HcomAllReduce, kNameAllReduce, ADPT_DESC(HcomAllReduce))

// HCOMBraodcast
INPUT_MAP(HcomBroadcast) = EMPTY_INPUT_MAP;
DYN_INPUT_MAP(HcomBroadcast) = {{1, DYN_INPUT_DESC(x)}};
DYN_OUTPUT_MAP(HcomBroadcast) = {{0, DYN_OUTPUT_DESC(y)}};
ATTR_MAP(HcomBroadcast) = {{"root_rank", ATTR_DESC(root_rank, AnyTraits<int64_t>())},
                           {"group", ATTR_DESC(group, AnyTraits<std::string>())}};
REG_ADPT_DESC(HcomBroadcast, kNameBroadcast, ADPT_DESC(HcomBroadcast))

// HcomAllGather
INPUT_MAP(HcomAllGather) = {{1, INPUT_DESC(x)}};
OUTPUT_MAP(HcomAllGather) = {{0, OUTPUT_DESC(y)}};
ATTR_MAP(HcomAllGather) = {{"group", ATTR_DESC(group, AnyTraits<std::string>())},
                           {"rank_size", ATTR_DESC(rank_size, AnyTraits<int64_t>())}};
REG_ADPT_DESC(HcomAllGather, kNameAllgather, ADPT_DESC(HcomAllGather))

// HCOMReduceScatter
INPUT_MAP(HcomReduceScatter) = {{1, INPUT_DESC(x)}};
OUTPUT_MAP(HcomReduceScatter) = {{0, OUTPUT_DESC(y)}};
ATTR_MAP(HcomReduceScatter) = {{"group", ATTR_DESC(group, AnyTraits<std::string>())},
                               {"op", ATTR_DESC(reduction, AnyTraits<std::string>())},
                               {"rank_size", ATTR_DESC(rank_size, AnyTraits<int64_t>())}};
REG_ADPT_DESC(HcomReduceScatter, kNameReduceScatter, ADPT_DESC(HcomReduceScatter))

INPUT_MAP(HcomSend) = {{1, INPUT_DESC(x)}};
ATTR_MAP(HcomSend) = {{"group", ATTR_DESC(group, AnyTraits<std::string>())},
                      {"sr_tag", ATTR_DESC(sr_tag, AnyTraits<int64_t>())},
                      {"dest_rank", ATTR_DESC(dest_rank, AnyTraits<int64_t>())}};
REG_ADPT_DESC(HcomSend, kNameSend, ADPT_DESC(HcomSend));

INPUT_MAP(HcomReceive) = EMPTY_INPUT_MAP;
OUTPUT_MAP(HcomReceive) = {{0, OUTPUT_DESC(y)}};
ATTR_MAP(HcomReceive) = {{"group", ATTR_DESC(group, AnyTraits<std::string>())},
                         {"sr_tag", ATTR_DESC(sr_tag, AnyTraits<int64_t>())},
                         {"src_rank", ATTR_DESC(src_rank, AnyTraits<int64_t>())},
                         {"shape", ATTR_DESC(shape, AnyTraits<std::vector<int64_t>>())},
                         {"dtype", ATTR_DESC(dtype, AnyTraits<GEType>())}};
REG_ADPT_DESC(HcomReceive, kNameReceive, ADPT_DESC(HcomReceive));
}  // namespace mindspore::transform
