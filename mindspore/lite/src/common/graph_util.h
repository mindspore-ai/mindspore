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

#ifndef MINDSPORE_LITE_SRC_COMMON_GRAPH_UTIL_H_
#define MINDSPORE_LITE_SRC_COMMON_GRAPH_UTIL_H_

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <memory>
#include "schema/model_generated.h"
#include "src/common/log_adapter.h"
#include "include/errorcode.h"
#include "include/model.h"

namespace mindspore {
namespace lite {
using NODE_ID = std::string;

std::vector<size_t> GetGraphInputNodes(const lite::Model *model);

std::vector<size_t> GetGraphOutputNodes(const lite::Model *model);

std::vector<size_t> GetLinkedPostNodeIdx(const lite::Model *model, size_t tensor_idx);

bool IsPackedOp(int op_type);
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_COMMON_GRAPH_UTIL_H_
