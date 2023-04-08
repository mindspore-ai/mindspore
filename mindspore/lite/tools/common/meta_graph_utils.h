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

#ifndef MINDSPORE_LITE_TOOLS_COMMON_META_GRAPH_UTILS_H_
#define MINDSPORE_LITE_TOOLS_COMMON_META_GRAPH_UTILS_H_

#include <vector>
#include "inner/model_generated.h"
#include "include/errorcode.h"
#include "src/common/log_util.h"
namespace mindspore::lite {
std::vector<size_t> GetLinkedPreIdx(const schema::MetaGraphT &graphT, const size_t &tensorIdx);

std::vector<size_t> GetLinkedPostIdx(const schema::MetaGraphT &graphT, const size_t &tensorIdx);

std::vector<size_t> GetInputNodeIdx(const schema::MetaGraphT &graphT, const schema::CNodeT &node,
                                    int inputIndexIdx = -1);

std::vector<size_t> GetInputNodeIdx(const schema::MetaGraphT &graphT, const size_t &nodeIdx, int inputIndexIdx = -1);

std::vector<size_t> GetOutputNodeIdx(const schema::MetaGraphT &graphT, const schema::CNodeT &node,
                                     int outputIndexIdx = -1);

std::vector<size_t> GetOutputNodeIdx(const schema::MetaGraphT &graphT, const size_t &nodeIdx, int outputIndexIdx = -1);

STATUS IsolateNode(schema::MetaGraphT *subGraph, schema::CNodeT *node);

STATUS RemoveTensor(schema::MetaGraphT *graphT, std::vector<uint32_t> toDeleteTensorIdxes, bool forceDelete = false);

void ReplaceOutput(const uint32_t &old_index, const uint32_t &new_index, schema::MetaGraphT *graphT);

STATUS UpdateNodeIndex(schema::CNodeT *node, uint32_t deleteIdx);

STATUS IsolateOneWayNode(schema::MetaGraphT *graphT, size_t subGraphIdx, size_t nodeIdx, bool removeTensor = true);

STATUS IsolateOneWayNode(schema::MetaGraphT *graphT, schema::CNodeT *node, bool removeTensor = true);

STATUS IsolateOneWayNode(schema::MetaGraphT *graphT, size_t nodeIdx, bool removeTensor = true);

}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_TOOLS_COMMON_META_GRAPH_UTILS_H_
